from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large as torch_lraspp_mobilenet_v3_large,
)
from tqdm import tqdm

from modeling import lraspp_mobilenet_v3_large


def main():
    weights_path = Path("weights/lraspp_mobilenet_v3_large.safetensors")
    mlx_model = lraspp_mobilenet_v3_large(num_classes=21).eval()
    mlx_model.load_weights(str(weights_path))
    mx.eval(mlx_model.parameters())

    torch_model = torch_lraspp_mobilenet_v3_large(
        weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    ).eval()

    mlx_input = mx.random.normal(
        shape=(1, 512, 512, 3), key=mx.array([42, 42], dtype=mx.uint32)
    )
    mlx_output = np.array(mlx_model(mlx_input).transpose(0, 3, 1, 2))

    torch_input = torch.from_numpy(np.asarray(mlx_input)).permute(0, 3, 1, 2)

    with torch.inference_mode():
        torch_output = torch_model(torch_input)["out"].numpy()

    print(f"Mean absolute error: {np.mean(np.abs(mlx_output - torch_output))}")
    print(f"Max absolute error: {np.max(np.abs(mlx_output - torch_output))}")

    assert np.allclose(mlx_output, torch_output, atol=1.0e-5)

    torch_model = torch_model.eval().to(device="mps")
    torch_input = torch_input.to(device="mps")
    with torch.inference_mode():
        for _ in tqdm(range(200), desc="PyTorch"):
            _ = torch_model(torch_input)

    mlx_model = mlx_model.eval()

    @mx.compile
    def inference_step(input_: mx.array) -> mx.array:
        return mlx_model(input_)

    # NOTE: Warmup
    mx.eval(inference_step(mlx_input))

    for _ in tqdm(range(200), desc="MLX"):
        output = inference_step(mlx_input)
        mx.eval(output)


if __name__ == "__main__":
    main()
