from .minimax_trt_node import MiniMaxH3ONNXVAELoader, MiniMaxH3TRTVAELoader, MiniMaxH3TRTCompilerNode

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ONNXVAELoader": MiniMaxH3ONNXVAELoader,
    "MiniMaxH3TRTVAELoader": MiniMaxH3TRTVAELoader,
    "MiniMaxH3TRTCompilerNode": MiniMaxH3TRTCompilerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ONNXVAELoader": "MiniMax-H3 ONNX VAE Loader",
    "MiniMaxH3TRTVAELoader": "MiniMax-H3 TRT VAE Loader",
    "MiniMaxH3TRTCompilerNode": "MiniMax-H3 TRT Compiler"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]