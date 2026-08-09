"""Person 4 model adapters (lazy imports where heavy deps are optional)."""

from adapters.unet_keras import UnetKerasAdapter
from adapters.segformer_stub import SegFormerStubAdapter, SegFormerNotReadyError
from adapters.deeplabv3 import DeepLabV3Adapter

__all__ = [
    "UnetKerasAdapter",
    "SegFormerStubAdapter",
    "SegFormerNotReadyError",
    "SegFormerAdapter",
    "DeepLabV3Adapter",
]


def __getattr__(name: str):
    # Lazy: transformers / Dinura attention_consistency only needed for SegFormer
    if name == "SegFormerAdapter":
        from adapters.segformer import SegFormerAdapter

        return SegFormerAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
