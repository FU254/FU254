#!/usr/bin/env python3
"""Encode the official MedKLIP disease book once, using an entirely local BERT."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def _save_book(payload, output):
    import torch

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".medklip-book-", dir=output.parent)
    try:
        # Torch 2.0's filename writer rejects some hidden temporary names.
        # Writing to the open binary stream works on both Linux and Windows.
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Refuse replacing a concurrently created output as well.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(book, text_encoder, output):
    import torch
    from transformers import BertModel, BertTokenizer
    from foundation_sft.models.common import sha256_file
    from foundation_sft.models.medklip import BOOK_PROTOCOL, BOOK_SHA256

    book, text_encoder, output = (Path(p).expanduser().resolve() for p in (book, text_encoder, output))
    if output.exists():
        raise ValueError("Output already exists; keep it or choose a fresh embedding path")
    if not book.is_file() or sha256_file(book) != BOOK_SHA256:
        raise ValueError("Use the reviewed official MedKLIP observation explanation JSON")
    data = json.loads(book.read_text(encoding="utf-8"))
    if len(data) != 75 or not all(isinstance(text, str) for text in data.values()):
        raise ValueError("Expected the ordered 75-query official disease book")
    if not text_encoder.is_dir():
        raise FileNotFoundError("Download emilyalsentzer/Bio_ClinicalBERT to --text-encoder first")
    names = ("config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json",
             "pytorch_model.bin", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    files = {p.name: sha256_file(p) for p in text_encoder.iterdir()
             if p.is_file() and (p.name in names or p.name.startswith(("model-", "pytorch_model-")))}
    if "config.json" not in files or "vocab.txt" not in files:
        raise ValueError("Local ClinicalBERT config.json and vocab.txt are required")
    # Like the author code, let the downloaded tokenizer configuration decide
    # casing. Do not impose ImageNet-like or uncased defaults on a text model.
    tokenizer = BertTokenizer.from_pretrained(str(text_encoder), local_files_only=True)
    model, info = BertModel.from_pretrained(str(text_encoder), local_files_only=True,
                                           add_pooling_layer=False, output_loading_info=True,
                                           attn_implementation="eager")
    unexpected = [key for key in info.get("unexpected_keys", [])
                  if not key.startswith(("pooler.", "cls."))]
    if unexpected or any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("ClinicalBERT backbone did not load completely")
    if (model.config.model_type, model.config.hidden_size, model.config.num_hidden_layers) != ("bert", 768, 12):
        raise ValueError("Use the official 768-channel 12-layer Bio_ClinicalBERT release")
    model.eval()
    tokens = tokenizer(list(data.values()), padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    with torch.no_grad():
        embeddings = torch.cat([model(**{key: value[start:start+8] for key, value in tokens.items()})
                                .last_hidden_state[:, 0].cpu().float() for start in range(0, 75, 8)])
    if not torch.isfinite(embeddings).all():
        raise ValueError("ClinicalBERT produced non-finite book embeddings")
    payload = {"embeddings": embeddings, "names": list(data), "book_sha256": BOOK_SHA256,
               "protocol": BOOK_PROTOCOL, "text_encoder_source": "emilyalsentzer/Bio_ClinicalBERT",
               "tokenizer_do_lower_case": bool(tokenizer.do_lower_case),
               "text_encoder_files_sha256": files}
    _save_book(payload, output)
    print("Prepared local MedKLIP disease embeddings:", output, flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default="foundation_weights/medklip/observation_explanation.json")
    parser.add_argument("--text-encoder", default="foundation_weights/medklip/clinicalbert")
    parser.add_argument("--output", default="foundation_weights/medklip/disease_book_embeddings.pt")
    args = parser.parse_args(argv)
    prepare(args.book, args.text_encoder, args.output)


if __name__ == "__main__":
    main()
