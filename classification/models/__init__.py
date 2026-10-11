from .models_eva import *

try:
    from .models_vit import *
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.models_vit":
        raise
