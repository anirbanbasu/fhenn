import torch
import tenseal as ts

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class CNN2D(torch.nn.Module):
    """A rudimentary CNN model for classification of MNIST-like 2D image data."""

    def __init__(self, hidden=64, output=10, device=None):
        """
        Initialize the CNN model.

        Args:
            hidden (int): The number of neurons in the hidden layer.
            output (int): The number of output neurons.
            device (str): The device to use for the model.
        """
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
            """ torch.device: The device to use for the model. """

        super(CNN2D, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=7,
            padding=0,
            stride=3,
            device=self.device,
        )
        """ torch.nn.Conv2d: The first convolutional layer. """

        self.fc1 = torch.nn.Linear(256, hidden, device=self.device)
        """ torch.nn.Linear: The first fully connected layer. """

        self.fc2 = torch.nn.Linear(hidden, output, device=self.device)
        """ torch.nn.Linear: The second fully connected layer. """

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        # square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x


class EncryptedCNN2D:
    """
    A class to represent an encrypted version of a CNN2D model.
    The CKKS fully homomorphic cryptosystem is used. [Research paper](https://eprint.iacr.org/2016/421.pdf)

    """

    def __init__(self, conv_net: CNN2D):
        """
        Initialize the encrypted CNN model.

        Args:
            conv_net (CNN2D): The CNN model to encrypt.
        """
        self.conv1_weight = conv_net.conv1.weight.data.view(
            conv_net.conv1.out_channels,
            conv_net.conv1.kernel_size[0],
            conv_net.conv1.kernel_size[1],
        ).tolist()
        """ list: The weights of the first convolutional layer. """
        self.conv1_bias = conv_net.conv1.bias.data.tolist()
        """ list: The biases of the first convolutional layer. """

        self.fc1_weight = conv_net.fc1.weight.T.data.tolist()
        """ list: The weights of the first fully connected layer. """
        self.fc1_bias = conv_net.fc1.bias.data.tolist()
        """ list: The biases of the first fully connected layer. """

        self.fc2_weight = conv_net.fc2.weight.T.data.tolist()
        """ list: The weights of the second fully connected layer. """
        self.fc2_bias = conv_net.fc2.bias.data.tolist()
        """ list: The biases of the second fully connected layer. """

    def forward(self, enc_x, windows_nb):
        """
        Forward pass of the encrypted model.

        Args:
            enc_x (tenseal.CKKSVector): The encrypted input tensor.
            windows_nb (int): The number of windows to use for the computation.

        Returns:
            tenseal.CKKSVector: The encrypted output tensor.
        """
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
        """
        Calls the forward pass of the encrypted model.

        Args:
            *args: The positional arguments.
            **kwargs: The keyword arguments.

        Returns:
            tenseal.CKKSVector: The encrypted output tensor.
        """
        return self.forward(*args, **kwargs)
