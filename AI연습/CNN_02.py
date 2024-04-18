import torch
import torch.nn as nn
import torch.optim as optim


class ConvolutionNeuralNetworkClass(nn.Module):
    def __init__(
        self,
        name="cnn",
        xdim=[1, 28, 28],
        ksize=3,
        cdims=[32, 64],
        hdims=[1024, 128],
        ydim=10,
        USE_BATCHNORM=False,
    ):
        super(ConvolutionNeuralNetworkClass, self).__init__()
        self.name = name
        self.xdim = xdim
        self.ksize = ksize
        self.cdims = cdims
        self.hdims = hdims
        self.ydim = ydim
        self.USE_BATCHNORM = USE_BATCHNORM

        self.layers = []
        prev_cdim = self.xdim[0]
        for cdim in self.cdims:
            self.layers.append(
                nn.Conv2d(
                    in_channels=prev_cdim,
                    out_channels=cdim,
                    kernel_size=self.ksize,
                    padding=(self.ksize // 2),
                )
            )
            if self.USE_BATCHNORM:
                self.layers.append(nn.BatchNorm2d(cdim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            self.layers.append(nn.Dropout2d(p=0.5))
            prev_cdim = cdim

        self.layers.append(nn.Flatten())
        prev_hdim = (
            prev_cdim
            * (self.xdim[1] // (2 ** len(self.cdims)))
            * (self.xdim[2] // (2 ** len(self.cdims)))
        )

        for hdim in self.hdims:
            self.layers.append(nn.Linear(prev_hdim, hdim))
            self.layers.append(nn.ReLU(inplace=True))
            prev_hdim = hdim

        self.layers.append(nn.Linear(prev_hdim, self.ydim, bias=True))

        self.net = nn.Sequential()
        for l_idx, layer in enumerate(self.layers):
            layer_name = "%s_%02d" % (type(layer).__name__.lower(), l_idx)
            self.net.add_module(layer_name, layer)
        self.init_param()

    def init_param(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
