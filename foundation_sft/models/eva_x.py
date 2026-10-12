"""EVA-X adapter using this repository's encoder and its corrected attention."""

from pathlib import Path

from .common import (BackboneAdapter, BackboneFeatures, PreprocessSpec,
                     load_checkpoint, load_backbone_weights, sha256_file)


class EVAXAdapter(BackboneAdapter):
    def __init__(self, backbone, provenance, pooling="patch_mean"):
        super().__init__()
        if pooling not in {"patch_mean", "legacy_all_tokens_mean"}:
            raise ValueError("EVA-X pooling must be patch_mean or legacy_all_tokens_mean")
        self.backbone = backbone
        self.feature_dim = backbone.embed_dim
        self.preprocess = PreprocessSpec(image_size=224, resize_mode="letterbox")
        self.pooling = pooling
        self.provenance = {**provenance, "pooling": pooling,
                           "local_features": "normalized spatial patches without CLS"}

    def forward_features(self, images):
        x = self.backbone.patch_embed(images)
        batch = x.shape[0]
        cls = self.backbone.cls_token.expand(batch, -1, -1)
        import torch
        x = torch.cat((cls, x), dim=1)
        if self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        rel_bias = self.backbone.rel_pos_bias() if self.backbone.rel_pos_bias is not None else None
        for block in self.backbone.blocks:
            x = block(x, rel_bias)
        x = self.backbone.norm(x)
        spatial = x[:, 1:]
        pooled = spatial.mean(1) if self.pooling == "patch_mean" else x.mean(1)
        if self.backbone.fc_norm is not None:
            pooled = self.backbone.fc_norm(pooled)
            spatial = self.backbone.fc_norm(spatial)
        patch = self.backbone.patch_embed.patch_size
        if isinstance(patch, int):
            patch = (patch, patch)
        grid = (images.shape[-2] // patch[0], images.shape[-1] // patch[1])
        return BackboneFeatures(pooled, spatial, grid).validate()

    def lora_target_names(self):
        return [f"backbone.blocks.{index}.attn.{name}"
                for index in range(len(self.backbone.blocks))
                for name in ("q_proj", "k_proj", "v_proj")]


def build_eva_x(config):
    from classification.models.models_eva import eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE
    path = Path(config.get("checkpoint", "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Set checkpoint to your eva_x_base_patch16_merged520k_mim.pt file")
    state = load_checkpoint(path, config.get("state_key"))
    normalization = _normalization_layout(state)
    backbone = eva02_base_patch16_xattn_fusedLN_NaiveSwiGLU_subln_RoPE(
        pretrained=False, num_classes=0, use_mean_pooling=normalization == "pooled_norm")
    # MIM mask/decoder and downstream classification heads are unused by the
    # feature encoder. Every other tensor, including its actual pretrained norm,
    # must match; norm before token pooling is not interchangeable with fc_norm.
    unused_keys = {"mask_token", "lm_head.weight", "lm_head.bias", "head.weight", "head.bias"}
    ignored = sorted(set(state) & unused_keys)
    encoder_state = {key: value for key, value in state.items() if key not in unused_keys}
    load_backbone_weights(backbone, encoder_state, ignored_prefixes=())
    return EVAXAdapter(backbone, {"model": "eva_x_base_patch16", "checkpoint": str(path.resolve()),
                                 "sha256": sha256_file(path),
                                 "normalization_layout": normalization,
                                 "ignored_checkpoint_keys": ignored,
                                 "source": "https://huggingface.co/MapleF/eva_x"},
                       pooling=config.get("pooling", "patch_mean"))


def _normalization_layout(state):
    """Keep the checkpoint's token norm or pooled norm at its original position."""
    layouts = []
    for prefix, layout in (("norm", "token_norm"), ("fc_norm", "pooled_norm")):
        required = {f"{prefix}.weight", f"{prefix}.bias"}
        present = required & set(state)
        if present and present != required:
            raise ValueError(f"Incomplete EVA-X {prefix} normalization weights: missing {sorted(required - present)}")
        if present:
            layouts.append(layout)
    if len(layouts) != 1:
        raise ValueError("EVA-X checkpoint must contain exactly one complete normalization pair: "
                         "norm.weight/norm.bias or fc_norm.weight/fc_norm.bias")
    return layouts[0]
