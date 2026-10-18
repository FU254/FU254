"""MedKLIP visual memory and knowledge-query decoder, with local book embeddings.

Architecture reviewed against MediaBrain-SJTU/MedKLIP revision
8e60b035ddc160f44a389f476b596f1052f02d8f, PreTrain_MedKLIP/models/
{model_MedKLIP,transformer}.py. This is a supervised downstream feature adapter,
not the author's 75-disease zero-shot classifier. Spatial memory and semantic
query tokens remain distinct: disease queries are never presented as a grid.
"""

import json

import torch
from torch import nn
from torch.nn import functional as F

from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, load_checkpoint, load_checkpoint_payload
from .extra_vlm_common import local_asset, require_linear_probe, strict_tensors


SOURCE_REVISION = "8e60b035ddc160f44a389f476b596f1052f02d8f"
SOURCE = "MediaBrain-SJTU/MedKLIP:resnet50"
BOOK_SHA256 = "ae872105337c7463e35826a3f726afba12455e45acbb4241c8e888d66f717b97"
BOOK_PROTOCOL = "Bio_ClinicalBERT_cls_max128_release_tokenizer_v1"


class KnowledgeDecoderLayer(nn.Module):
    """The released pre-norm self/cross attention layer; state names preserved."""

    def __init__(self, dim=256, heads=4, hidden=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.linear1, self.linear2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.dropout = nn.Dropout(dropout)
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(dropout) for _ in range(3))

    def forward(self, queries, memory):
        normalized = self.norm1(queries)
        update = self.self_attn(normalized, normalized, normalized, need_weights=True)[0]
        queries = queries + self.dropout1(update)
        update = self.multihead_attn(self.norm2(queries), memory, memory, need_weights=True)[0]
        queries = queries + self.dropout2(update)
        update = self.linear2(self.dropout(F.relu(self.linear1(self.norm3(queries)))))
        return queries + self.dropout3(update)


class KnowledgeDecoder(nn.Module):
    def __init__(self, dim=256, heads=4, layers=4, hidden=1024):
        super().__init__()
        self.layers = nn.ModuleList([KnowledgeDecoderLayer(dim, heads, hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, queries, memory):
        for layer in self.layers:
            queries = layer(queries, memory)
        return self.norm(queries)


class MedKLIPVisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50

        resnet = resnet50(weights=None)
        self.res_features = nn.Sequential(*list(resnet.children())[:-3])  # through layer3, 1024 channels
        self.res_l1, self.res_l2 = nn.Linear(1024, 1024), nn.Linear(1024, 256)
        self.disease_embedding_layer = nn.Linear(768, 256)
        self.cl_fc = nn.Linear(256, 768)  # retained for strict checkpoint completeness
        self.decoder = KnowledgeDecoder()
        self.dropout_feas = nn.Dropout(0.1)
        self.classifier = nn.Linear(256, 2)  # original disease-attribute head, unused downstream

    def features(self, images, book):
        feature_map = self.res_features(images)
        grid = tuple(int(x) for x in feature_map.shape[-2:])
        memory = feature_map.flatten(2).transpose(1, 2)
        # Preserve the author's 2-D GEMM ordering rather than applying Linear
        # directly to a non-contiguous 3-D tensor (different GPU/CPU kernels).
        batch, count, channels = memory.shape
        flat = memory.reshape(batch * count, channels)
        memory = self.res_l2(F.relu(self.res_l1(flat))).reshape(batch, count, -1)
        queries = self.disease_embedding_layer(book).unsqueeze(1).repeat(1, batch, 1)
        queries = self.decoder(queries, memory.transpose(0, 1)).transpose(0, 1)
        return memory, queries, grid


def load_book(config):
    book_path, book_hash = local_asset(config, "disease_book")
    if book_hash != BOOK_SHA256:
        raise ValueError("MedKLIP disease book differs from the reviewed official ordered 75-query book")
    names = list(json.loads(book_path.read_text(encoding="utf-8")))
    path, checksum = local_asset(config, "disease_embeddings")
    payload = load_checkpoint_payload(path)
    if (payload.get("book_sha256") != book_hash or payload.get("names") != names
            or payload.get("protocol") != BOOK_PROTOCOL
            or payload.get("text_encoder_source") != "emilyalsentzer/Bio_ClinicalBERT"
            or type(payload.get("tokenizer_do_lower_case")) is not bool
            or not isinstance(payload.get("text_encoder_files_sha256"), dict)
            or not payload["text_encoder_files_sha256"]):
        raise ValueError("MedKLIP book embedding identity/order/protocol is invalid; run prepare_medklip_book.py")
    embeddings = payload.get("embeddings")
    if (not isinstance(embeddings, torch.Tensor) or embeddings.shape != (75, 768)
            or embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all()):
        raise ValueError("MedKLIP requires finite float32 [75,768] ClinicalBERT book embeddings")
    return embeddings, {"disease_book": str(book_path), "disease_book_sha256": book_hash,
                        "disease_embeddings": str(path), "disease_embeddings_sha256": checksum,
                        "book_embedding_protocol": BOOK_PROTOCOL,
                        "tokenizer_do_lower_case": payload["tokenizer_do_lower_case"],
                        "text_encoder_files_sha256": payload["text_encoder_files_sha256"]}


class MedKLIPAdapter(BackboneAdapter):
    def __init__(self, visual, book, preprocess, provenance):
        super().__init__()
        self.visual = visual
        self.register_buffer("disease_book", book, persistent=False)
        self.feature_dim = 256
        self.preprocess, self.provenance = preprocess, provenance

    def lora_target_names(self):
        raise ValueError("MedKLIP currently supports linear_probe; no decoder/CNN LoRA equivalence is claimed")

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.preprocess.image_size, self.preprocess.image_size):
            raise ValueError("MedKLIP expects recorded square RGB inputs")
        memory, queries, grid = self.visual.features(images, self.disease_book)
        return BackboneFeatures(queries.mean(dim=1), memory, grid).validate()

    def forward_query_features(self, images):
        """Optional semantic teacher tokens [B,75,256]; these have no grid."""
        return self.visual.features(images, self.disease_book)[1]


def build_medklip(config):
    require_linear_probe(config)
    if config.get("checkpoint_source") != SOURCE:
        raise ValueError(f"MedKLIP requires checkpoint_source={SOURCE!r}")
    path, checksum = local_asset(config, "checkpoint")
    book, book_metadata = load_book(config)
    visual = MedKLIPVisualModel()
    strict_tensors(visual, load_checkpoint(path, state_key="model"), "MedKLIP full visual/query model")
    preprocess = PreprocessSpec(224, resize_mode="resize", interpolation="bilinear")
    return MedKLIPAdapter(visual, book, preprocess, {
        "model": "medklip", "official_repository": "https://github.com/MediaBrain-SJTU/MedKLIP",
        "verified_source_revision": SOURCE_REVISION, "checkpoint_source": SOURCE,
        "checkpoint": str(path), "checkpoint_sha256": checksum, "state_key": "model",
        "architecture": "ResNet50 through layer3 + 1024-to-256 visual projection + four-layer knowledge-query decoder",
        "global_features": "Research downstream choice: mean over 75 final normalized disease-query tokens",
        "local_features": "Spatial image memory after res_l1/ReLU/res_l2, BEFORE the query decoder",
        "semantic_query_features": "forward_query_features returns [B,75,256]; query axis is NOT a spatial grid",
        "unused_pretraining_heads": "Original attribute classifier and anatomy-contrastive cl_fc retained and strictly loaded",
        "preprocessing_implementation": "shared_dicom_display_then_medklip_resize224_bilinear_imagenet_v1",
        **book_metadata,
    })
