"""Model imports are lazy so each encoder can use its own supported environment."""

SUPPORTED_MODELS = ("eva_x", "rad_dino", "ray_dino", "chexfound", "cxformer", "radjepa")


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
    raise ValueError(f"Unknown model {name!r}; choose from {SUPPORTED_MODELS}")
