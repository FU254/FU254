"""Optional add-on registry; importing this module does not import Torch."""

EXTRA_MODELS = ("mgca", "biovil_t", "medklip")


def build_extra_adapter(config):
    name = config["model"]
    if name == "mgca":
        from .mgca import build_mgca
        return build_mgca(config)
    if name == "biovil_t":
        from .biovil_t import build_biovil_t
        return build_biovil_t(config)
    if name == "medklip":
        from .medklip import build_medklip
        return build_medklip(config)
    raise ValueError("Unknown extra VLM encoder")
