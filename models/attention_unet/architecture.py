import torch
import torch.nn as nn

class Attention_UNet_architecture(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, init_features = 32):
        super(Attention_UNet_architecture, self).__init__()

        features = init_features

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, features)
        self.encoder2 = conv_block(features, features * 2)
        self.encoder3 = conv_block(features * 2, features * 4)
        self.encoder4 = conv_block(features * 4, features * 8)
        self.encoder5 = conv_block(features * 8, features * 16)

        self.up5 = up_conv(features * 16, features * 8)
        self.attention5 = attention_block(F_g=features * 8, F_l=features * 8, F_int=features * 4)
        self.up_conv5 = conv_block(features * 16, features * 8)

        self.up4 = up_conv(features * 8, features * 4)
        self.attention4 = attention_block(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.up_conv4 = conv_block(features * 8, features * 4)

        self.up3 = up_conv(features * 4, features * 2)
        self.attention3 = attention_block(F_g=features * 2, F_l=features * 2, F_int=features)
        self.up_conv3 = conv_block(features * 4, features * 2)
        
        self.up2 = up_conv(features * 2, features)
        self.attention2 = attention_block(F_g=features, F_l=features, F_int=features//2)
        self.up_conv2 = conv_block(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        en1 = self.encoder1(x)
        en2 = self.encoder2(self.maxpool(en1))
        en3 = self.encoder3(self.maxpool(en2))
        en4 = self.encoder4(self.maxpool(en3))

        bottleneck = self.encoder5(self.maxpool(en4))

        de5 = self.up5(bottleneck)
        en4 = self.attention5(de5, en4)
        de5 = torch.cat((de5, en4), dim=1)
        de5 = self.up_conv5(de5)

        de4 = self.up4(de5)
        en3 = self.attention4(de4, en3)
        de4 = torch.cat((de4, en3), dim=1)
        de4 = self.up_conv4(de4)

        de3 = self.up3(de4)
        en2 = self.attention3(de3, en2)
        de3 = torch.cat((de3, en2), dim=1)
        de3 = self.up_conv3(de3)

        de2 = self.up2(de3)
        en1 = self.attention2(de2, en1)
        de2 = torch.cat((de2, en1), dim=1)
        de2 = self.up_conv2(de2)

        out = self.conv(de2)
        return out

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi