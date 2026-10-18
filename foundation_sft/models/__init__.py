"""Model imports are lazy so each encoder can use its own supported environment."""

try:
    from .extra_vlm_registry import EXTRA_MODELS
except ModuleNotFoundError as error:
    if error.name != __name__ + ".extra_vlm_registry":
        raise
    # Earlier scoped CNN/VLM installers also ship this lazy registry. They do
    # not advertise additional encoders unless the complete new add-on exists.
    EXTRA_MODELS = ()

SUPPORTED_MODELS = ("eva_x", "rad_dino", "ray_dino", "chexfound", "cxformer", "radjepa",
                    "cxr_clip", "xray_siglip", "biomedclip", "chexzero",
                    "densenet121_imagenet", "densenet121_cxr", "resnet18_cxr") + EXTRA_MODELS


def build_adapter(config):
    name = config["model"]
    if name in EXTRA_MODELS:
        from .extra_vlm_registry import build_extra_adapter
        return build_extra_adapter(config)
    if name in {"densenet121_imagenet", "densenet121_cxr", "resnet18_cxr"}:
        from .cnn import build_cnn
        return build_cnn(config)
    if name == "eva_x":
        from .eva_x import build_eva_x
        return build_eva_x(config)
    if name == "ray_dino" and config.get("backend", "native") == "native":
        from .ray_dino import build_ray_dino
        return build_ray_dino(config)
    if name in {"rad_dino", "ray_dino"}:
        from .hf_dino import build_hf_dino
        return build_hf_dino(config)
    if name == "chexfound":
        from .chexfound import build_chexfound
        return build_chexfound(config)
    if name == "cxformer":
        from .cxformer import build_cxformer
        return build_cxformer(config)
    if name == "radjepa":
        from .radjepa import build_radjepa
        return build_radjepa(config)
    if name == "cxr_clip":
        from .cxr_clip import build_cxr_clip
        return build_cxr_clip(config)
    if name == "xray_siglip":
        from .hf_siglip import build_hf_siglip
        return build_hf_siglip(config)
    if name == "biomedclip":
        from .openclip_vlm import build_openclip_vlm
        return build_openclip_vlm(config)
    if name == "chexzero":
        from .chexzero import build_chexzero
        return build_chexzero(config)
    raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")
