import torch.nn as nn

def get_blocks(num_layers):
    if num_layers == 50:
        return [
            [{'in_channels': 64, 'depth': 64, 'stride': 1}],
            [{'in_channels': 64, 'depth': 128, 'stride': 2}],
            [{'in_channels': 128, 'depth': 256, 'stride': 2}],
            [{'in_channels': 256, 'depth': 512, 'stride': 2}]
        ]

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channels, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        self.shortcut_layer = nn.Sequential()
        if stride == 2:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, depth, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channels, depth, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth)
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(depth, depth // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth // 16, depth, kernel_size=1),
            nn.Sigmoid()
        )
        self.prelu = nn.PReLU(depth)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        res = res * self.se(res)
        res += shortcut
        return self.prelu(res)
