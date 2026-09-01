import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.cli_args
import comfy.model_management as mm
from server import PromptServer

try:
    import tensorrt as trt
    HAS_TRT = True
except ImportError:
    HAS_TRT = False

logger = logging.getLogger(__name__)

# ================= 1. 检测 ComfyUI 启动参数与全局生命周期 Hook =================

IS_HIGH_VRAM = getattr(comfy.cli_args.args, "gpu_only", False) or getattr(comfy.cli_args.args, "highvram", False)

ACTIVE_TRT_RUNNERS = set()

def offload_all_trt_runners():
    for runner in list(ACTIVE_TRT_RUNNERS):
        runner.offload_to_ram()

if not hasattr(mm, "_minimax_trt_hook_applied"):
    mm._minimax_trt_hook_applied = True

    orig_unload_all_models = mm.unload_all_models
    def patched_unload_all_models(*args, **kwargs):
        offload_all_trt_runners()
        return orig_unload_all_models(*args, **kwargs)
    mm.unload_all_models = patched_unload_all_models

    if hasattr(mm, "free_memory"):
        orig_free_memory = mm.free_memory
        def patched_free_memory(*args, **kwargs):
            offload_all_trt_runners()
            return orig_free_memory(*args, **kwargs)
        mm.free_memory = patched_free_memory


# ================= 2. 内存/显存引擎执行器 =================

class AutoEngineRunner:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.stream = None
        self.engine_bytes = None
        self.runtime = None
        self.engine = None
        self.context = None
        self.session = None
        ACTIVE_TRT_RUNNERS.add(self)

    def load_to_ram(self):
        if self.engine_bytes is None:
            with open(self.model_path, "rb") as f:
                self.engine_bytes = f.read()

    def load_to_gpu(self):
        if self.context is not None or self.session is not None:
            return

        self.load_to_ram()
        if not HAS_TRT:
            raise RuntimeError("TensorRT library not found!")
        if self.runtime is None:
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_bytes)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def offload_to_ram(self):
        if self.context is not None:
            self.context = None
            self.engine = None
            self.stream = None
            torch.cuda.empty_cache()
        if self.session is not None:
            self.session = None
            torch.cuda.empty_cache()

    def infer(self, input_tensor: torch.Tensor, output_shape: tuple, input_name: str) -> torch.Tensor:
        # 1. 确保引擎已加载到显存
        self.load_to_gpu()
        
        device = input_tensor.device
        dtype = input_tensor.dtype
        input_tensor = input_tensor.contiguous()
        
        if self.stream is None or self.stream.device != device:
            self.stream = torch.cuda.Stream(device=device)
            
        self.stream.wait_stream(torch.cuda.current_stream(device))
        
        # 分配输出张量内存（用 zeros 避免未初始化脏显存泄露）
        output = torch.zeros(output_shape, dtype=dtype, device=device)
        self.context.set_input_shape(input_name, input_tensor.shape)
        self.context.set_tensor_address(input_name, input_tensor.data_ptr())
        out_name = "pixel_tile" if input_name == "latent_tile" else "moments_tile"
        self.context.set_tensor_address(out_name, output.data_ptr())
        
        # 🌟 传入非默认独立流，彻底消除 TensorRT 的 enqueueV3() 性能警告
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output

    def __del__(self):
        ACTIVE_TRT_RUNNERS.discard(self)


# ================= 3. MiniMax-H3 加速 VAE 实现 =================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775, 2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265, 3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525]


class MiniMaxH3AcceleratedVAE(nn.Module):
    def __init__(self, decoder_runner: AutoEngineRunner = None, encoder_runner: AutoEngineRunner = None):
        super().__init__()
        self.vae_dtype = torch.float16
        self.decoder_runner = decoder_runner
        self.encoder_runner = encoder_runner

        self.vae_ratio = 16
        self.vae_ratio_t = 4
        self.clip_length = 17
        self.token_drop = 3
        self.frame_pre_padding = (-self.clip_length) % self.vae_ratio_t  # 3
        self.tokens_chunk_size = math.ceil(self.clip_length / self.vae_ratio_t)  # 5
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size  # 2
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)  # 8

        self.tile_size = 256
        self.tile_overlap_min = 64

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        
    def _decode_temporal_chunks(self, z_len):
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens
        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1
        return pad_tokens, num_chunks

    # 🌟 2. 补上官方对齐的单块切片解码别名
    def _adaptive_decode(self, z):
        return self.tiled_decode(z) 

    def _decode_temporal_pad_frames(self, z_len, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        
        z_len_before_pad = z_len - pad_tokens
        return sum(
                intra_tail
                if (z_len_before_pad + k) % self.tokens_chunk_size == 0
                else self.vae_ratio_t
                for k in range(pad_tokens)
        )
            
    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0
        
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t
            
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames
                    
        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)
        
    # 🌟 补齐双卡流式拼接所需的输出尺寸计算函数
    def decode_output_shape(self, input_shape):
        b, c, t, h, w = input_shape
        if t == 1:
            frames = 1
        else:
            pad_tokens, num_chunks = self._decode_temporal_chunks(t)
            frames = self._decode_temporal_frame_plan(t + pad_tokens, num_chunks, pad_tokens)
        return (b, 3, frames, h * self.vae_ratio, w * self.vae_ratio)

    def _decode_pixels(self, z):
        b, _, t, h, w = z.shape
        out_shape = (b, 3, t * self.vae_ratio_t, h * self.vae_ratio, w * self.vae_ratio)
        return self.decoder_runner.infer(z, output_shape=out_shape, input_name="latent_tile")

    def _encode_moments(self, x):
        b, c, t, h, w = x.shape
        target_h, target_w = self.tile_size, self.tile_size
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)
        
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0.0)
        else:
            x_in = x
            
        out_shape = (b, 48, math.ceil(t / self.vae_ratio_t), target_h // self.vae_ratio, target_w // self.vae_ratio,)
        moments = self.encoder_runner.infer(x_in, output_shape=out_shape, input_name="pixel_tile")
        
        out_h = math.ceil(h / self.vae_ratio)
        out_w = math.ceil(w / self.vae_ratio)
        return moments[..., :out_h, :out_w]

    def _finalize_pixels(self, part):
        return (part * self.pixel_std.to(part) + self.pixel_mean.to(part)).clamp(0.0, 1.0)

    def _normalize_pixels(self, x):
        return (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        if blend_extent <= 0:
            return b

        weight = torch.arange(blend_extent, device=b.device, dtype=b.dtype) / blend_extent
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight = weight.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = torch.lerp(a[tuple(slice_a)], b[tuple(slice_b)], weight)
        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def split_tiles(self, input_len):
        if self.tile_size >= input_len:
            return [0], [input_len], []
        N = math.ceil(input_len / self.tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = self.tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break
        for i in range(remaining // self.vae_ratio):
            overlaps[i % (N - 1)] += self.vae_ratio
        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + self.tile_size - overlaps[i])
        return tile_start_idx, [self.tile_size] * N, overlaps

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        canvas, row_tails, out_y = None, [], 0
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len)):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            new_tails, left_tail, out_x = [], None, 0
            for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len)):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                tile = self._decode_pixels(z[..., zi:zi + zl, zj:zj + zw])

                if i < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[i]:, :].clone())
                next_left_tail = tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None

                if i > 0:
                    tile = self.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
                left_tail = next_left_tail

                if i < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[i], :]
                if j < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[j]]

                if canvas is None:
                    canvas = torch.zeros(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y:out_y + tile.shape[-2], out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)

        rows = []
        for i_pos, i_len in zip(y_idx, y_len):
            row = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len]
                row.append(self._encode_moments(tile))
            rows.append(row)

        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def decode_temporal(self, z):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        pseudo_total_tokens = z.shape[2] + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        num_chunks = (pseudo_total_tokens + pad_tokens) // self.tokens_chunk_size - 1

        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        dec_chunks, dec_overlap = [], None
        for i in range(num_chunks):
            t_start = i * self.tokens_chunk_size
            clip_z = z[:, :, t_start:t_start + self.tokens_chunk_size + self.token_overlap, :, :]
            clip_dec = self.tiled_decode(clip_z)

            for j in range(2):
                f_start = j * chunk_dec
                chunk = clip_dec[:, :, f_start + self.frame_pre_padding:min(f_start + chunk_dec, clip_dec.shape[2]), :, :]
                if j == 0:
                    if dec_overlap is not None:
                        chunk = self.blend(dec_overlap, chunk, self.frame_overlap, dim=-3)
                        dec_overlap = None
                    dec_chunks.append(self._finalize_pixels(chunk))
                else:
                    dec_overlap = chunk.contiguous()

            if i == num_chunks - 1 and dec_overlap is not None:
                dec_chunks.append(self._finalize_pixels(dec_overlap))

        return torch.cat(dec_chunks, dim=2)

    def encode_temporal(self, x):
        z_list = []
        num_clips = math.ceil(x.shape[2] / self.clip_length)
        for i in range(num_clips):
            clip_x = x[:, :, i * self.clip_length:(i + 1) * self.clip_length, :, :]
            if clip_x.shape[2] < self.clip_length:
                pad_frames = clip_x[:, :, -1:].repeat(1, 1, self.clip_length - clip_x.shape[2], 1, 1)
                clip_x = torch.cat([clip_x, pad_frames], dim=2)
            z_list.append(self.tiled_encode(self._normalize_pixels(clip_x)))

        z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, :-self.token_drop]
        return z

    def decode(self, z):
        if self.decoder_runner is None:
            raise RuntimeError("Decoder model is not loaded!")
        try:
            z = z * self.latents_std.to(z) + self.latents_mean.to(z)
            if z.shape[2] == 1:
                return self._finalize_pixels(self.tiled_decode(z)[:, :, -1:, :, :])
            return self.decode_temporal(z)
        finally:
            if not IS_HIGH_VRAM and self.decoder_runner is not None:
                self.decoder_runner.offload_to_ram()

    def encode(self, x):
        if self.encoder_runner is None:
            raise RuntimeError("Encoder model is not loaded!")
        try:
            if x.ndim == 4:
                x = x.unsqueeze(2)
            if x.shape[2] == 1:
                moments = self.tiled_encode(self._normalize_pixels(x))[:, :, -1:, :, :]
            else:
                moments = self.encode_temporal(x)

            mean = torch.chunk(moments, 2, dim=1)[0]
            return (mean - self.latents_mean.to(mean)) / self.latents_std.to(mean)
        finally:
            if not IS_HIGH_VRAM and self.encoder_runner is not None:
                self.encoder_runner.offload_to_ram()


# ================= 4. ComfyUI 标准节点包装 =================

class ComfyVAEWrapper:
    def __init__(self, first_stage_model):
        self.trt = True
        self.first_stage_model = first_stage_model

    def decode(self, samples_in):
        z = samples_in["samples"] if isinstance(samples_in, dict) else samples_in
        if z.ndim == 4:
            z = z.unsqueeze(2)
        video = self.first_stage_model.decode(z.half().cuda())
        b, c, t, h, w = video.shape
        return video.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).float().cpu()

    def encode(self, pixel_in):
        if pixel_in.ndim == 4:
            x = pixel_in.permute(3, 0, 1, 2).unsqueeze(0)
        elif pixel_in.ndim == 5:
            x = pixel_in.permute(0, 4, 1, 2, 3)
        else:
            raise ValueError(f"Unsupported pixel tensor shape: {pixel_in.shape}")

        latents = self.first_stage_model.encode(x.half().cuda())
        return latents.float().cpu()

class MiniMaxH3TRTVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for path in folder_paths.get_folder_paths("vae"):
            for root, _, fs in os.walk(path):
                for f in fs:
                    if f.endswith(".engine"):
                        files.append(os.path.relpath(os.path.join(root, f), path))
        files = sorted(list(dict.fromkeys(files)))
        options = ["None"] + files
        
        return {
            "required": {
                "decoder": (options, {
                    "default": "None",
                    "tooltip": 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.'
                }),
                "encoder": (options, {
                    "default": "None",
                    "tooltip": 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.'
                }),
            }
        }
    
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "MiniMax_H3/Acceleration"
    DESCRIPTION = 'Please compile the TensorRT engine using the "MiniMax-H3 TRT VAE Compiler" node before first use.'
    
    def load_vae(self, decoder, encoder):
        if encoder == "None":
            raise RuntimeError("Encoder cannot be None!")
        if decoder == "None":
            raise RuntimeError("Decoder cannot be None!")
            
        dec_path = folder_paths.get_full_path("vae", decoder)
        enc_path = folder_paths.get_full_path("vae", encoder)
        
        dec_runner = AutoEngineRunner(dec_path) if dec_path else None
        enc_runner = AutoEngineRunner(enc_path) if enc_path else None
        
        vae_instance = MiniMaxH3AcceleratedVAE(decoder_runner=dec_runner, encoder_runner=enc_runner)
        return (ComfyVAEWrapper(vae_instance),)

class MiniMaxH3TRTCompilerNode:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        for path in folder_paths.get_folder_paths("vae"):
            for root, _, fs in os.walk(path):
                for f in fs:
                    if f.endswith(".onnx"):
                        files.append(os.path.relpath(os.path.join(root, f), path))
        files = sorted(list(dict.fromkeys(files)))
        options = ["None"] + files
        
        return {
            "required": {
                "decoder_onnx": (options,),
                "encoder_onnx": (options,),
                "delete_onnx_after_compile": ("BOOLEAN",{
                    "default": False,
                    "tooltip": "Delete ONNX weights from disk after compilation."
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "compile_models"
    CATEGORY = "MiniMax_H3/Acceleration"
    
    @classmethod
    def _build_engine(cls, onnx_path, engine_path, is_decoder=True):
        if not HAS_TRT:
            raise RuntimeError("TensorRT library not found! Please install with pip first.")
            
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        
        # 🌟 兼容 TRT 8.x / 10.x / 11.x 的 Network 创建
        if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(flags)
        else:
            network = builder.create_network()
            
        parser = trt.OnnxParser(network, logger)
        
        # 🌟 使用 parse_from_file，确保正确读取同目录下的 .onnx.data
        if not parser.parse_from_file(onnx_path):
            error_msgs = []
            for error in range(parser.num_errors):
                error_msgs.append(str(parser.get_error(error)))
            raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(error_msgs))
            
        # 🌟 兼容 TRT 8.x ~ 10.x（TRT 11.x 彻底删除了 FP16 弱类型 Flag）
        if hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
            
        # 设置工作区显存 (Decoder 分配 4GB, Encoder 分配 8GB)
        workspace_size = (4 if is_decoder else 8) * (1024**3)
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(
                    trt.MemoryPoolType.WORKSPACE, workspace_size
            )
        else:
            config.max_workspace_size = workspace_size
            
        # 设置标准 Profile 尺寸
        profile = builder.create_optimization_profile()
        if is_decoder:
            input_name = "latent_tile"
            shape = (1, 24, 7, 16, 16)
        else:
            input_name = "pixel_tile"
            shape = (1, 3, 17, 256, 256)
            
        profile.set_shape(input_name, shape, shape, shape)
        config.add_optimization_profile(profile)
        
        # 开始构建并保存到同目录下同名文件
        if hasattr(builder, "build_serialized_network"):
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build the TensorRT engine!")
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build the TensorRT engine!")
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
                
    def compile_models(self, decoder_onnx, encoder_onnx, delete_onnx_after_compile, unique_id):
        dec_path = folder_paths.get_full_path("vae", decoder_onnx)
        enc_path = folder_paths.get_full_path("vae", encoder_onnx)
        
        if dec_path is None and enc_path is None:
            raise RuntimeError("No ONNX file selected!")
        
        # 1. 编译前清空显存，防止显存爆炸
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        
        # 2. 编译 Decoder
        if dec_path is not None:
            engine_path = os.path.splitext(dec_path)[0] + ".engine"
            logger.info(f"Building Decoder: {os.path.basename(dec_path)}...")
            self._build_engine(dec_path, engine_path, is_decoder=True)
            logger.info(f"✅ Done: {os.path.basename(engine_path)}")
            if delete_onnx_after_compile:
                os.remove(dec_path)
            
        # 3. 编译 Encoder
        if enc_path is not None:
            engine_path = os.path.splitext(enc_path)[0] + ".engine"
            logger.info(f"Building Encoder: {os.path.basename(enc_path)}...")
            self._build_engine(enc_path, engine_path, is_decoder=False)
            logger.info(f"✅ Done: {os.path.basename(engine_path)}")
            if delete_onnx_after_compile:
                os.remove(enc_path)
            
        # 4. 完成提示
        logger.info('🎉 All Done! Please press "R" to refresh the model list.')
        PromptServer.instance.send_progress_text('Done! Press "R" to refresh the model list.', unique_id)
        return ()