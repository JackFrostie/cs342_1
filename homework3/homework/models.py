import sys
import torch
import torch.nn.functional as F
import torchvision.transforms
from math import log

START_SIZE_AND_DEPTH1 = [16, 3]
INPUTNORM1 = True
RANDCROP1 = False
RANDFLIP1 = True
RANDCOLOR1 = True
DROPOUT1 = True
# CHNLNORM = False
STATAUG1 = False
ZRO1 = True
ADAM1 = True
PATIENCE1 = 5
BRIGHTNESS1 = 3
CONTRAST1 = 1.5
EPOCHS1 = 1
BS1 = 512
LR1 = 0.0005


DROP_P = 0.2


INPUTNORM2 = True
RANDCROP2 = False
RANDFLIP2 = True
RANDCOLOR2 = True
DROPOUT2 = True
# CHNLNORM = False
STATAUG2 = True
ZRO2 = False
ADAM2 = True
PATIENCE2 = 5
BRIGHTNESS2 = 3
CONTRAST2 = 1.5
EPOCHS2 = 50
BS2 = 64
LR2 = 0.005
DEPTH = 4
START_SIZE = 16



class ConvBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, dropout=False):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_input, out_channels=n_output, kernel_size=3, padding=1, bias=False),
                                       torch.nn.BatchNorm2d(n_output))
        if dropout:
            self.net.append(torch.nn.Dropout2d(p=DROP_P))
        self.net.append(torch.nn.ReLU(inplace=True))
        self.net.append(torch.nn.Conv2d(in_channels=n_output, out_channels=n_output, kernel_size=3, padding=1, bias=False))
        self.net.append(torch.nn.BatchNorm2d(n_output))
        if dropout:
            self.net.append(torch.nn.Dropout2d(p=DROP_P))
        self.net.append(torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)

class DownBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, dropout=False):
        super().__init__()
        self.net1 = torch.nn.Sequential(ConvBlock(n_input=n_input, n_output=n_output, dropout=dropout),
                                       torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.downsample = None
        if n_input != n_output:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels=n_input, out_channels=n_output, kernel_size=1, stride=2),
                                                  torch.nn.BatchNorm2d(n_output))
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        return self.net1(x) + identity

class UpBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, dropout=False):
        super().__init__()
        self.net2 = torch.nn.ConvTranspose2d(in_channels=n_input, out_channels=n_input//2, kernel_size=3, stride=2, padding=1)
        self.net3 = ConvBlock(n_input=n_input, n_output=n_output, dropout=dropout)

    def forward(self, x_in, x_cat):
        x_in = self.net2(x_in)

        y_dim = x_cat.size()[2] - x_in.size()[2]
        x_dim = x_cat.size()[3] - x_in.size()[3]
        x_in = F.pad(x_in, [x_dim//2, x_dim - x_dim//2,
                            y_dim//2, y_dim-y_dim//2])

        x = torch.cat([x_cat, x_in], dim=1)
        return self.net3(x)


def get_stats(data):
    dmean = torch.zeros([1, 3])
    tempsum = torch.zeros([1, 3])
    for dat, label in data:
        dmean += dat.mean(dim=[2, 3])

    mn = dmean / data.__len__()

    for dat, label in data:
        tempsum += ((dat - mn[:, :, None, None]).square()).mean(dim=[2, 3])

    stddv = (tempsum / data.__len__()).sqrt()

    return mn.squeeze().tolist(), stddv.squeeze().tolist()


# def channel_normalize(x):
#     x = (x - x.mean(dim=[0, 2, 3], keepdim=False)[None, :, None, None]) / (
#                 x.std(dim=[0, 2, 3], keepdim=False)[None, :, None, None] + 1e-8)
#     return x



class CNNClassifier(torch.nn.Module):

    def __init__(self, n_input_channels=3, dropout=False, stat_data=None):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        layers = START_SIZE_AND_DEPTH1
        # layers = [4, 3]

        # self.mean, self.std = get_stats(stat_data)

        # The beginning of the network will cut down the feature plane twice
        #   which will make it 1/4th of what it was originally.
        # This is because of the stride in the convolution and the stride in the maxpool layer.

        L = [torch.nn.Conv2d(in_channels=n_input_channels, out_channels=layers[0], kernel_size=7, padding=3, stride=2, bias=False),
             torch.nn.BatchNorm2d(layers[0])]
        if dropout:
            L.append(torch.nn.Dropout2d(p=DROP_P))
        L.append(torch.nn.ReLU(inplace=True))
        L.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        c = layers[0]
        for _ in range(layers[1]):
            L.append(DownBlock(n_input=c, n_output=c*2, dropout=dropout))
            c *= 2

        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, 6)
        if ZRO1:
            torch.nn.init.zeros_(self.classifier.weight)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        """
        # Normalize the channels across the batch. This should be roughly ok if BS is large enough.

        # tform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=self.mean, std=self.std)])
        if INPUTNORM1:
            tform = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.32352423667907715, 0.33100518584251404, 0.34449565410614014],
                                                                                     std=[0.25328782200813293, 0.22241945564746857, 0.24833780527114868])])
            x = tform(x)

        z = self.network(x)
        # Global averaging to reduce the dimension to a vector so that it can be put into linear
        z = z.mean(dim=[2, 3])

        # Return logits
        return self.classifier(z)


class FCN(torch.nn.Module):
    def __init__(self, n_input_size=3, stat_data=None, dropout=DROPOUT2):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        self.start_size = START_SIZE
        self.l1 = ConvBlock(n_input_size, self.start_size, dropout=DROPOUT2)

        a = self.start_size * 2
        b = a * 2
        c = b * 2
        d = c * 2

        if 0 < DEPTH:
            self.down1 = DownBlock(self.start_size, self.start_size*2, dropout=False)
        if 1 < DEPTH:
            self.down2 = DownBlock(a, b, dropout=False)
        if 2 < DEPTH:
            self.down3 = DownBlock(b, c, dropout=False)
        if 3 < DEPTH:
            self.down4 = DownBlock(c, d, dropout=False)
        if 3 < DEPTH:
            self.up4 = UpBlock(d, c, dropout=False)
        if 2 < DEPTH:
            self.up3 = UpBlock(c, b, dropout=False)
        if 1 < DEPTH:
            self.up2 = UpBlock(b, a, dropout=False)
        if 0 < DEPTH:
            self.up1 = UpBlock(a, self.start_size, dropout=False)

        self.classifier = torch.nn.Conv2d(self.start_size, 5, kernel_size=1)
        if ZRO2:
            torch.nn.init.zeros_(self.classifier.weight)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        if INPUTNORM2:
            tform = torchvision.transforms.Compose(
                [torchvision.transforms.Normalize(mean=[0.2788424789905548, 0.2657163143157959, 0.26285597681999207],
                                                  std=[0.2064191848039627, 0.19443656504154205, 0.22521907091140747])])
            x = tform(x)

        if DEPTH == 4:
            x1 = self.l1(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up4(x5, x4)
            x = self.up3(x, x3)
            x = self.up2(x, x2)
            x = self.up1(x, x1)
        if DEPTH == 3:
            x1 = self.l1(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x = self.up3(x4, x3)
            x = self.up2(x, x2)
            x = self.up1(x, x1)

        if DEPTH == 2:
            x1 = self.l1(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up2(x3, x2)
            x = self.up1(x, x1)

        if DEPTH == 1:
            x1 = self.l1(x)
            x2 = self.down1(x1)
            x = self.up1(x2, x1)

        logits = self.classifier(x)
        return logits



model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
