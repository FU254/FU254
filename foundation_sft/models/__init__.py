"""Model imports are lazy so each encoder can use its own supported environment."""

SUPPORTED_MODELS = ("eva_x", "rad_dino", "ray_dino", "chexfound", "cxformer", "radjepa",
                    "cxr_clip", "xray_siglip", "biomedclip", "chexzero")


def build_adapter(config):
    name = config["model"]
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
    if name in {"xray_siglip", "biomedclip"}:
        from .openclip_vlm import build_openclip_vlm
        return build_openclip_vlm(config)
    if name == "chexzero":
        from .chexzero import build_chexzero
        return build_chexzero(config)
    raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")
