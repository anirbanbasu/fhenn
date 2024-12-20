import torch
import tenseal as ts

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class ConvNet(torch.nn.Module):
    def __init__(self, hidden=64, output=10, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                if torch.backends.mps.is_built():
                    self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            1, 4, kernel_size=7, padding=0, stride=3, device=self.device
        )
        self.fc1 = torch.nn.Linear(256, hidden, device=self.device)
        self.fc2 = torch.nn.Linear(hidden, output, device=self.device)

    def forward(self, x):
        x = self.conv1(x)
        # square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


class EncConvNet:
    def __init__(self, conv_net: ConvNet):
        self.conv1_weight = conv_net.conv1.weight.data.view(
            conv_net.conv1.out_channels,
            conv_net.conv1.kernel_size[0],
            conv_net.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = conv_net.conv1.bias.data.tolist()

        self.fc1_weight = conv_net.fc1.weight.T.data.tolist()
        self.fc1_bias = conv_net.fc1.bias.data.tolist()

        self.fc2_weight = conv_net.fc2.weight.T.data.tolist()
        self.fc2_bias = conv_net.fc2.bias.data.tolist()

    def forward(self, enc_x, windows_nb):
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
