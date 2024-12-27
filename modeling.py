import warnings
from collections import abc
from functools import partial
from itertools import repeat
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
from mlx import nn


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    if isinstance(x, abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input: mx.array) -> mx.array:
        return mx.clip(input / 6.0 + 0.5, 0.0, 1.0)


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: Union[int, Tuple[int, int]] = 1) -> None:
        super().__init__()
        self.output_size = (
            output_size
            if isinstance(output_size, tuple)
            else (output_size, output_size)
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = x.reshape(
            B,
            H // self.output_size[0],
            self.output_size[0],
            W // self.output_size[1],
            self.output_size[1],
            C,
        )
        return mx.mean(x, axis=(1, 3))


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)
                )
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {}
            layers.append(activation_layer(**params))

        super().__init__(*layers)
        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation and Conv3dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            bias,
            nn.Conv2d,
        )


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: mx.array) -> mx.array:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def __call__(self, input: mx.array) -> mx.array:
        scale = self._scale(input)
        return scale * input


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(
            SqueezeExcitation,
            scale_activation=HardSigmoid,
        ),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def __call__(self, input: mx.array) -> mx.array:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3Backbone(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all(
                [
                    isinstance(s, InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm, eps=0.001, momentum=0.01)

        self.layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            self.layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels

        self.layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

    def __call__(self, x: mx.array) -> List[mx.array]:
        return [x := m(x) for m in self.layers]


class LRASPPHead(nn.Module):
    def __init__(
        self,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int,
    ) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear")

        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm(inter_channels),
            nn.ReLU(),
        )
        self.scale = nn.Sequential(
            AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def __call__(self, low: mx.array, high: mx.array) -> mx.array:
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = self.upsample(x)

        return self.low_classifier(low) + self.high_classifier(x)


class LRASPP(nn.Module):
    def __init__(
        self,
        backbone: MobileNetV3Backbone,
        low_channels: int,
        high_channels: int,
        num_classes: int,
        inter_channels: int = 128,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels
        )
        self.upsample = nn.Upsample(scale_factor=8, mode="linear")

    def __call__(self, input: mx.array) -> mx.array:
        features = self.backbone(input)
        out = self.classifier(features[4], features[-1])
        return self.upsample(out)


def mobilenet_v3_large_backbone():
    bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160, True, "HS", 2, 2),  # C4
        bneck_conf(160, 5, 960, 160, True, "HS", 1, 2),
        bneck_conf(160, 5, 960, 160, True, "HS", 1, 2),
    ]

    return MobileNetV3Backbone(inverted_residual_setting)


def lraspp_mobilenet_v3_large(num_classes: int) -> LRASPP:
    backbone = mobilenet_v3_large_backbone()

    return LRASPP(
        backbone=backbone,
        low_channels=40,
        high_channels=960,
        num_classes=num_classes,
    )
