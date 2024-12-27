import re
from itertools import starmap
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import torch
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)


def map_torch_to_mlx(
    key: str, value: torch.Tensor
) -> tuple[str, mx.array] | tuple[None, None]:
    def replace(matchobj: re.Match) -> str:
        return f".layers{matchobj.group(0)}"

    key = re.sub(r"\.[0-9]+", replace, key)

    if key.endswith("weight") and value.dim() == 4:
        value = value.permute(0, 2, 3, 1)
    elif key.endswith("num_batches_tracked"):
        return None, None

    return key, mx.array(value.numpy())


def convert_weights(state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: v for k, v in starmap(map_torch_to_mlx, state.items()) if k is not None}


def main():
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)

    state = model.state_dict()
    weights = convert_weights(state)

    mx.save_safetensors(
        str(weights_dir / "lraspp_mobilenet_v3_large.safetensors"), arrays=weights
    )


if __name__ == "__main__":
    main()
