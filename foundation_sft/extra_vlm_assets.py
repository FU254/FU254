"""Author-published asset locations. Importable without Torch or Transformers."""

EXTRA_VLM_ASSETS = {
    "cxr_clip": {
        "filename": "cxr_clip/swint_mcc.tar",
        "url": "https://twg.kakaocdn.net/brainrepo/models/cxr-clip/a25ce760064591c899f67c97ed7790de/swint_mcc.tar",
        "source": "https://github.com/Soombit-ai/cxr-clip",
    },
    "mgca": {
        "filename": "mgca/mgca_vit_base.ckpt",
        "manual_url": "https://drive.google.com/drive/folders/15_mP9Lqq2H15R53qlKn3l_xzGVzi9jX9?usp=sharing",
        "source": "https://github.com/HKU-MedAI/MGCA",
        "instruction": "Choose the official ViT-Base MGCA Lightning checkpoint, not the ResNet checkpoint. Save/rename it to the configured filename.",
    },
    "biovil_t": {
        "filename": "biovil_t/biovil_t_image_model_proj_size_128.pt",
        "url": "https://huggingface.co/microsoft/BiomedVLP-BioViL-T/resolve/v1.0/biovil_t_image_model_proj_size_128.pt",
        "md5": "a83080e2f23aa584a4f2b24c39b1bb64",
        "source": "https://github.com/microsoft/hi-ml/tree/b67c1d27c6b17d8e8ff01f8c507f3cabdb307388/hi-ml-multimodal",
    },
    "medklip": {
        "filename": "medklip/checkpoint_final.pth",
        "manual_url": "https://pan.baidu.com/s/1qea5jVtHcW81i-jI4IP7Cg?pwd=2vuc",
        "alternative_url": "https://drive.google.com/drive/folders/1HBShH7J_pO8onkzuweDgDq2QPqj6zjG_?usp=sharing",
        "source": "https://github.com/MediaBrain-SJTU/MedKLIP/tree/8e60b035ddc160f44a389f476b596f1052f02d8f/checkpoints",
        "instruction": "Choose the complete official MedKLIP checkpoint with model state and knowledge-query decoder, not a converted ResNet backbone.",
    },
}

MEDKLIP_BOOK = {
    "filename": "medklip/observation_explanation.json",
    "url": "https://raw.githubusercontent.com/MediaBrain-SJTU/MedKLIP/8e60b035ddc160f44a389f476b596f1052f02d8f/PreTrain_MedKLIP/data_file/observation%20explanation.json",
    "sha256": "ae872105337c7463e35826a3f726afba12455e45acbb4241c8e888d66f717b97",
}
