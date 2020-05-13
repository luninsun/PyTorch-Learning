import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    """
    resnet basic block
    """

    def __init__(self, in_channels, out_channels, stride = 1):
        """

        :param in_channels:
        :param out_channels:
        :param stride:
        """
        super(BasicBlock, self).__init__()

        # add stride support for resblk
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        if in_channels != out_channels:
            self.downsample = nn.Sequential()
            self.downsample.add_module('downsample', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride))
        else:
            self.downsample = lambda x : x


    def forward(self, inputs):
        """
        """
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.downsample(inputs) + out
        out = F.relu(out, inplace=True)

        return out

class ResNet(nn.Module):

    def __init__(self, layer_dims, num_classes = 10):

        super(ResNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

        self.layer1 = self.build_resblock(64, 64, layer_dims[0])
        self.layer2 = self.build_resblock(64, 128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(128, 256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(256, 512, layer_dims[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*1*1, num_classes)

    def forward(self, inputs):

        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
    def build_resblock(self, in_channels, out_channels, blocks, stride=1):

        res_blocks = nn.Sequential()

        res_blocks.add_module('block0', BasicBlock(in_channels, out_channels, stride=stride))

        for i in range(1, blocks):
            res_blocks.add_module('block{}'.format(i), BasicBlock(out_channels, out_channels, stride=1))

        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])

def resnet34():
    return ResNet([3, 4, 6, 3])

def main():
    
    x = torch.randn(2, 3, 224, 224)
    model = resnet18()
    out = model(x)
    print(model)
    print(out.shape)

if __name__ == "__main__":
    main()