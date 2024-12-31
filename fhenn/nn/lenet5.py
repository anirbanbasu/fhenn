import torch
from torch import nn

from fhenn.constants import Constants


class LeNet5(nn.Module):
    """
    Reference implementation of the LeNet5 model.

    See: https://www.digitalocean.com/community/tutorials/writing-lenet5-from-scratch-in-python
    """

    def __init__(self, num_classes=10, device=None):
        """
        Initialize the LeNet5 model.

        Args:
            num_classes (int): The number of output classes.
            device (str): The device to use for the model
        """

        if device is None:
            if torch.backends.mps.is_available():
                if torch.backends.mps.is_built():
                    self.device = torch.device(Constants.DEVICE_LABEL_MPS)
            elif torch.cuda.is_available():
                self.device = torch.device(Constants.DEVICE_LABEL_CUDA)
            else:
                self.device = torch.device(Constants.DEVICE_LABEL_CPU)
        else:
            self.device = torch.device(device)
            """ torch.device: The device to use for the model. """

        super(LeNet5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, device=self.device),
            nn.BatchNorm2d(6, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """ The first layer group of the model. """

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, device=self.device),
            nn.BatchNorm2d(16, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        """ The second layer group of the model. """

        self.fc = nn.Linear(400, 120, device=self.device)
        """ The first fully connected layer. """

        self.relu = nn.ReLU()
        """ The first ReLU activation function. """

        self.fc1 = nn.Linear(120, 84, device=self.device)
        """ The second fully connected layer. """

        self.relu1 = nn.ReLU()
        """ The second ReLU activation function. """

        self.fc2 = nn.Linear(84, num_classes, device=self.device)
        """ The third fully connected layer. """

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out
