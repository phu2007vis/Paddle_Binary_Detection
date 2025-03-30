

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import MaxPool2D
from paddle import ParamAttr
import paddle.nn as nn
from paddle.regularizer import L2Decay
from paddle.nn.initializer import  Constant
from paddle.vision.ops import DeformConv2D

class DeformableConvV2(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        weight_attr=None,
        bias_attr=None,
        lr_scale=1,
        regularizer=None,
        skip_quant=False,
        dcn_bias_regularizer=L2Decay(0.0),
        dcn_bias_lr_scale=2.0,
    ):
        super(DeformableConvV2, self).__init__()
        self.offset_channel = 2 * kernel_size**2 * groups
        self.mask_channel = kernel_size**2 * groups

        if bias_attr:
            # in FCOS-DCN head, specifically need learning_rate and regularizer
            dcn_bias_attr = ParamAttr(
                initializer=Constant(value=0),
                regularizer=dcn_bias_regularizer,
                learning_rate=dcn_bias_lr_scale,
            )
        else:
            # in ResNet backbone, do not need bias
            dcn_bias_attr = False
        self.conv_dcn = DeformConv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation,
            deformable_groups=groups,
            weight_attr=weight_attr,
            bias_attr=dcn_bias_attr,
        )

        if lr_scale == 1 and regularizer is None:
            offset_bias_attr = ParamAttr(initializer=Constant(0.0))
        else:
            offset_bias_attr = ParamAttr(
                initializer=Constant(0.0),
                learning_rate=lr_scale,
                regularizer=regularizer,
            )
        self.conv_offset = nn.Conv2D(
            in_channels,
            groups * 3 * kernel_size**2,
            kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            weight_attr=ParamAttr(initializer=Constant(0.0)),
            bias_attr=offset_bias_attr,
        )
        if skip_quant:
            self.conv_offset.skip_quant = True

    def forward(self, x):
        offset_mask = self.conv_offset(x)
        offset, mask = paddle.split(
            offset_mask,
            num_or_sections=[self.offset_channel, self.mask_channel],
            axis=1,
        )
        mask = F.sigmoid(mask)
        y = self.conv_dcn(x, offset, mask=mask)
        return y

class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        dcn_groups=1,
        is_vd_mode=False,
        act=None,
        is_dcn=False,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )
        if not is_dcn:
            self._conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias_attr=False,
            )
        else:
            self._conv = DeformableConvV2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=dcn_groups,  # groups,
                bias_attr=False,
            )
        self._batch_norm = nn.BatchNorm(out_channels, act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y
    
class BottleneckBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True, is_dcn=False):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=1,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu",
            is_dcn=is_dcn,
            dcn_groups=1,
        )
        self.conv2 = ConvBNLayer(
            in_channels=num_filters,
            out_channels=num_filters * 4,
            kernel_size=1,
            act=None,
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters * 4,
                kernel_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

        self._num_channels_out = num_filters * 4

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y

class BasicBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, stride, shortcut=True, name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=stride,
            act="relu",
        )
        self.conv1 = ConvBNLayer(
            in_channels=num_filters, out_channels=num_filters, kernel_size=3, act=None
        )

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=num_channels,
                out_channels=num_filters,
                kernel_size=1,
                stride=stride,
            )

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = paddle.add(x=short, y=conv1)
        y = F.relu(y)
        return y

class ResNet(nn.Layer):
    def __init__(self, in_channels=3, layers=50, out_indices=None, dcn_stage=None,**kwargs):
        super(ResNet, self).__init__()

        self.layers = layers
        self.input_image_channel = in_channels

        supported_layers = [18, 34, 50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [64, 256, 512, 1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.dcn_stage = (
            dcn_stage if dcn_stage is not None else [False, False, False, False]
        )
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]

        self.conv = ConvBNLayer(
            in_channels=self.input_image_channel,
            out_channels=64,
            kernel_size=7,
            stride=2,
            act="relu",
        )
        self.pool2d_max = MaxPool2D(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.stages = []
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                is_dcn = self.dcn_stage[block]
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    bottleneck_block = self.add_sublayer(
                        conv_name,
                        BottleneckBlock(
                            num_channels=(
                                num_channels[block]
                                if i == 0
                                else num_filters[block] * 4
                            ),
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                            is_dcn=is_dcn,
                        ),
                    )
                    block_list.append(bottleneck_block)
                    shortcut = True
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block] * 4)
                self.stages.append(nn.Sequential(*block_list))
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    basic_block = self.add_sublayer(
                        conv_name,
                        BasicBlock(
                            num_channels=(
                                num_channels[block] if i == 0 else num_filters[block]
                            ),
                            num_filters=num_filters[block],
                            stride=2 if i == 0 and block != 0 else 1,
                            shortcut=shortcut,
                        ),
                    )
                    block_list.append(basic_block)
                    shortcut = True
                if block in self.out_indices:
                    self.out_channels.append(num_filters[block])
                self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.pool2d_max(y)
        out = []
        for i, block in enumerate(self.stages):
            y = block(y)
            if i in self.out_indices:
                out.append(y)
        return out
