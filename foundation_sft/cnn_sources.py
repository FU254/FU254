"""Verified public sources for the three frozen CNN baselines."""

CNN_SOURCES = {
    "densenet121_imagenet": {
        "architecture": "densenet121", "input_channels": 3,
        "checkpoint_name": "densenet121-a639ec97.pth",
        "url": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "sha256_prefix": "a639ec97",
        "sha256": "a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395",
        "repository": "https://github.com/pytorch/vision",
        "pretraining": "ImageNet-1K supervised; torchvision IMAGENET1K_V1",
        "preprocessing": {"image_size": 224, "resize_mode": "shortest_edge_center_crop",
                          "resize_shortest_edge": 256, "interpolation": "bilinear",
                          "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    },
    "densenet121_cxr": {
        "architecture": "densenet121", "input_channels": 1,
        "checkpoint_name": "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt",
        "url": "https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt",
        "sha256": "56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899",
        "repository": "https://github.com/mlmed/torchxrayvision",
        "pretraining": "TorchXRayVision densenet121-res224-all; multi-dataset CXR supervised, 18 pathology outputs",
        "preprocessing": {"image_size": 224, "resize_mode": "shortest_edge_center_crop",
                          "resize_shortest_edge": 224, "interpolation": "bilinear",
                          "mean": (0.5, 0.5, 0.5), "std": (1 / 2048, 1 / 2048, 1 / 2048)},
    },
    "resnet18_cxr": {
        "architecture": "resnet18", "input_channels": 3,
        "checkpoint_name": "r8w-00001.pth.tar",
        "url": "https://storage.googleapis.com/moco-cxr/r8w-00001.pth.tar",
        "repository": "https://github.com/stanfordmlgroup/MoCo-CXR",
        "pretraining": "MoCo-CXR ResNet18; ImageNet initialization followed by CheXpert CXR self-supervision",
        "preprocessing": {"image_size": 320, "resize_mode": "shortest_edge_center_crop",
                          "resize_shortest_edge": 320, "interpolation": "bilinear",
                          "mean": (0.5020, 0.5020, 0.5020), "std": (0.085585, 0.085585, 0.085585)},
    },
}
