
import torch
import torch.nn as nn

class DuckNetArchitecture(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, init_features = 17): # 17 is the default value in the paper, also they use 34
        super(DuckNetArchitecture, self).__init__()
        features = init_features

        self.conv1 = conv_block(in_channels, features*2, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv2 = conv_block(features*2, features*4, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv3 = conv_block(features*4, features*8, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv4 = conv_block(features*8, features*16, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv5 = conv_block(features*16, features*32, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)

        self.duck_down1 = duck_block(in_channels, features)
        self.duck_down2 = duck_block(features*2, features*2)
        self.duck_down3 = duck_block(features*4, features*4)
        self.duck_down4 = duck_block(features*8, features*8)
        self.duck_down5 = duck_block(features*16, features*16)

        self.conv_duck1 = conv_block(features, features*2, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv_duck2 = conv_block(features*2, features*4, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv_duck3 = conv_block(features*4, features*8, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv_duck4 = conv_block(features*8, features*16, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)
        self.conv_duck5 = conv_block(features*16, features*32, kernel_size=2, stride=2, padding=0, is_relu=False, is_norm=False)

        self.residual1 = residual_block(features*32, features*32)
        self.residual2 = residual_block(features*32, features*16)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.duck_up5 = duck_block(features*16, features*8)
        self.duck_up4 = duck_block(features*8, features*4)
        self.duck_up3 = duck_block(features*4, features*2)
        self.duck_up2 = duck_block(features*2, features)
        self.duck_up1 = duck_block(features, features)

        self.last_conv = nn.Conv2d(features, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        duck_down1 = self.duck_down1(x)
        conv_duck1 = self.conv_duck1(duck_down1)
        add1 = conv1 + conv_duck1

        duck_down2 = self.duck_down2(add1)
        conv_duck2 = self.conv_duck2(duck_down2)
        add2 = conv2 + conv_duck2

        duck_down3 = self.duck_down3(add2)
        conv_duck3 = self.conv_duck3(duck_down3)
        add3 = conv3 + conv_duck3

        duck_down4 = self.duck_down4(add3)
        conv_duck4 = self.conv_duck4(duck_down4)
        add4 = conv4 + conv_duck4

        duck_down5 = self.duck_down5(add4)
        conv_duck5 = self.conv_duck5(duck_down5)
        add5 = conv5 + conv_duck5

        residual1 = self.residual1(add5)
        residual2 = self.residual2(residual1)

        up5 = self.up5(residual2)
        add_up5 = duck_down5 + up5
        duck_up5 = self.duck_up5(add_up5)

        up4 = self.up4(duck_up5)
        add_up4 = duck_down4 + up4
        duck_up4 = self.duck_up4(add_up4)

        up3 = self.up3(duck_up4)
        add_up3 = duck_down3 + up3
        duck_up3 = self.duck_up3(add_up3)

        up2 = self.up2(duck_up3)
        add_up2 = duck_down2 + up2
        duck_up2 = self.duck_up2(add_up2)

        up1 = self.up1(duck_up2)
        add_up1 = duck_down1 + up1
        duck_up1 = self.duck_up1(add_up1)

        x = self.last_conv(duck_up1)

        # TODO: Research if this is the correct way to handle the output activation
        if self.last_conv.out_channels > 1:
            return torch.softmax(x, dim=1) # multi-class segmentation
        else:
            return torch.sigmoid(x) # binary segmentation

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, is_relu=True, is_norm=True):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True) if is_relu else None
        self.norm = nn.BatchNorm2d(out_channels) if is_norm else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(residual_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.shortcut = conv_block(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, is_relu=True, is_norm=False)

        self.residual_path = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation=1, is_relu=True, is_norm=True),
            conv_block(out_channels, out_channels, kernel_size, stride, padding, dilation=1, is_relu=True, is_norm=True)
        )

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.residual_path(x)
        x += shortcut
        x = self.norm(x)
        return x
    
class midscope_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(midscope_block, self).__init__()

        self.block = nn.Sequential(
            conv_block(input_channel, output_channel, kernel_size=3, stride=1, padding=1, dilation=1, is_relu=True, is_norm=True),
            conv_block(output_channel, output_channel, kernel_size=3, stride=1, padding=2, dilation=2, is_relu=True, is_norm=True)
        )

    def forward(self, x):
        return self.block(x)
    
class widescope_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(widescope_block, self).__init__()

        self.block = nn.Sequential(
            conv_block(input_channel, output_channel, kernel_size=3, stride=1, padding=1, dilation=1, is_relu=True, is_norm=True),
            conv_block(output_channel, output_channel, kernel_size=3, stride=1, padding=2, dilation=2, is_relu=True, is_norm=True),
            conv_block(output_channel, output_channel, kernel_size=3, stride=1, padding=3, dilation=3, is_relu=True, is_norm=True)
        )
    
    def forward(self, x):
        return self.block(x)

class separated_block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super(separated_block, self).__init__()

        self.block = nn.Sequential(
            conv_block(input_channel, output_channel, kernel_size=(3, kernel_size), stride=1, padding=1, dilation=1, is_relu=True, is_norm=True),
            conv_block(output_channel, output_channel, kernel_size=(kernel_size, 3), stride=1, padding=1, dilation=1, is_relu=True, is_norm=True)
        )

    def forward(self, x):
        return self.block(x)
    

class duck_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(duck_block, self).__init__()

        self.widescope_block = widescope_block(input_channel, output_channel)
        self.midscope_block = midscope_block(input_channel, output_channel)
        self.residual1 = residual_block(input_channel, output_channel)
        self.resifual2 = nn.Sequential(
            residual_block(input_channel, output_channel),
            residual_block(output_channel, output_channel)
        )
        self.resifual3 = nn.Sequential(
            residual_block(input_channel, output_channel),
            residual_block(output_channel, output_channel),
            residual_block(output_channel, output_channel)
        )
        self.separated_block = separated_block(input_channel, output_channel)

        self.norm = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        widescope = self.widescope_block(x)
        midscope = self.midscope_block(x)
        residual1 = self.residual1(x)
        residual2 = self.resifual2(x)
        residual3 = self.resifual3(x)
        separated = self.separated_block(x)
        x = widescope + midscope + residual1 + residual2 + residual3 + separated
        x = self.norm(x)
        return x