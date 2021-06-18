#!/usr/bin/env python3
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import onnx
# To avoid mac os x seg fault issue
import onnxruntime
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.imagenet import ImageNet

import furiosa_sdk_accuracy_debugger


def main():
    assets = Path(__file__).absolute().parent.parent / "assets"

    model = onnx.load(str(assets / "debug_ptq" / "mlcommons_resnet50_v1.5.onnx"))
    quantized_model = onnx.load(str(assets / "debug_ptq" / "mlcommons_resnet50_v1.5_fake_quant.onnx"))

    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(256, Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.485, 0.456, 0.406)),
                std=torch.tensor((0.229, 0.224, 0.225)),
            ),
        ]
    )
    dataset = ImageNet(str(assets / "imagenet"), "val", transform=imagenet_transform)

    # The shape of the model's input is [N, C, H, W] where N = 1.
    dataloader = DataLoader(dataset, batch_size=1)
    # The name of the model's input is `input.1`.
    calibration_dataset: List[Dict[str, np.ndarray]] = [
        {"input_tensor:0": x.numpy()} for x, _ in dataloader
    ]

    furiosa_sdk_accuracy_debugger.debug_ptq(
        model=model,
        validation_dataset=calibration_dataset,
        quantized_model=quantized_model,
        calibration_dataset=calibration_dataset,
        threshold=0.0,
    )


if __name__ == "__main__":
    sys.exit(main())
