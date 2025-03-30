import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr


class DSConv(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride=1,
        groups=None,
        if_act=True,
        act="relu",
        **kwargs,
    ):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )

        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )

        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False,
            )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x
    
class DBFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, use_asf=False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf
        weight_attr = paddle.nn.initializer.KaimingUniform()

        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse
    
class ASFBlock(nn.Layer):
    """
    This code is referred from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels, inter_channels, out_features_num=4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2D(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            # Nx1xHxW
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                bias_attr=False,
                padding=1,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=1,
                out_channels=1,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.Sigmoid(),
        )

        self.channel_scale = nn.Sequential(
            nn.Conv2D(
                in_channels=inter_channels,
                out_channels=out_features_num,
                kernel_size=1,
                bias_attr=False,
                weight_attr=ParamAttr(initializer=weight_attr),
            ),
            nn.Sigmoid(),
        )

    def forward(self, fuse_features, features_list):
        fuse_features = self.conv(fuse_features)
        spatial_x = paddle.mean(fuse_features, axis=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i : i + 1] * features_list[i])
        return paddle.concat(out_list, axis=1)
