"""El GPT Neural Network Engine Package."""

try:
    from .model import ElGPTConfig, ElGPTModel
    from .tokenizer import ElGPTTokenizer
    __all__ = ["ElGPTConfig", "ElGPTModel", "ElGPTTokenizer"]
except Exception:
    # Running in lightweight cloud/serverless environment without PyTorch
    __all__ = []
