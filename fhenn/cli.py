import numpy as np
import torch
from torchvision import datasets, transforms
import typer
import tenseal as ts
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from datetime import datetime
import warnings

from fhenn.constants import Constants
from fhenn.nn import ConvNet, EncConvNet

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

app = typer.Typer(
    name="FHENN",
    no_args_is_help=True,
    chain=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

# torch.manual_seed(73)

# print([dataset.id for dataset in list_datasets()])


@app.command(
    short_help="The de-facto Hello World command.",
    help="This command prints a slightly unusual Hello World message.",
    no_args_is_help=True,
)
def hello(name: str):
    console = Console()
    msg = Markdown(
        f"Hello _{name}_! This is a **slightly unusual** `Hello World` message.",
        justify="center",
    )
    console.print(
        Panel(
            msg,
            title=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            title_align="right",
            expand=False,
            padding=(1, 2),
        )
    )


def _train(
    model: torch.nn.Module, train_loader, criterion, optimizer, device, n_epochs=10
):
    model.train()
    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)

        print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch, train_loss))

    # model in evaluation mode
    model.eval()
    return model


@app.command(
    short_help="The training command",
    help="This command trains a simple convolutional neural network on the MNIST dataset.",
)
def train_model(
    model_path: str = typer.Option(
        help="The path to save the trained model.", writable=True
    ),
):
    train_data = datasets.MNIST(
        Constants.DATA_DIRECTORY,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    model = ConvNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = _train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        device=model.device,
        optimizer=optimizer,
        n_epochs=25,
    )
    torch.save(model.state_dict(), model_path)


def _test(model, test_loader, criterion, device):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    # model in evaluation mode
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


@app.command(
    short_help="The plaintext test command.",
    help="This command performs test on a previously trained model.",
)
def plaintext_test_model(
    model_path: str = typer.Option(
        help="The path to the trained model.", readable=True, exists=True
    ),
):
    test_data = datasets.MNIST(
        Constants.DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )
    model = ConvNet()
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()
    _test(
        model=model, test_loader=test_loader, criterion=criterion, device=model.device
    )


def _enc_test(context: ts.context, model, test_loader, criterion, kernel_shape, stride):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))

    for data, target in test_loader:
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context,
            data.view(28, 28).tolist(),
            kernel_shape[0],
            kernel_shape[1],
            stride,
        )
        # Encrypted evaluation
        enc_model = EncConvNet(model)
        enc_output = enc_model(x_enc, windows_nb)
        # Decryption of result
        output = enc_output.decrypt()
        output = torch.tensor(output).view(1, -1)

        # compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        label = target.data[0]
        class_correct[label] += correct.item()
        class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    print(f"Test Loss: {test_loss:.6f}\n")

    for label in range(10):
        print(
            f"Test Accuracy of {label}: {int(100 * class_correct[label] / class_total[label])}% "
            f"({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})"
        )

    print(
        f"\nTest Accuracy (Overall): {int(100 * np.sum(class_correct) / np.sum(class_total))}% "
        f"({int(np.sum(class_correct))}/{int(np.sum(class_total))})"
    )


@app.command(
    short_help="The encrypted test command.",
    help="This command performs encrypted test on a previously trained model.",
)
def encrypted_test_model(
    model_path: str = typer.Option(
        help="The path to the trained model.", readable=True, exists=True
    ),
):
    test_data = datasets.MNIST(
        Constants.DATA_DIRECTORY,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )
    model = ConvNet(device="cpu")
    model.load_state_dict(torch.load(model_path))
    criterion = torch.nn.CrossEntropyLoss()
    kernel_shape = model.conv1.kernel_size
    stride = model.conv1.stride[0]

    # controls precision of the fractional part
    bits_scale = 26

    # Create TenSEAL context
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[
            31,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            bits_scale,
            31,
        ],
    )

    # set the scale
    context.global_scale = pow(2, bits_scale)

    # galois keys are required to do ciphertext rotations
    context.generate_galois_keys()

    _enc_test(
        context=context,
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        kernel_shape=kernel_shape,
        stride=stride,
    )


@app.command(
    short_help="The key generation command",
    help="This command generates keys for the CKKS scheme.",
)
def keygen():
    console = Console()
    msg = Markdown(
        "Generating keys for the CKKS scheme...",
        justify="center",
    )
    console.print(
        Panel(
            msg,
            title=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            title_align="right",
            expand=False,
            padding=(1, 2),
        )
    )
    context = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    console.print(f"Public Key: {context.public_key()}")
    console.print(f"Secret Key: {context.secret_key()}")
