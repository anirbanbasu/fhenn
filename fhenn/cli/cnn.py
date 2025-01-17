from enum import Enum
from typing import Optional
import numpy as np
import torch
from torchvision import datasets, transforms
import typer
import tenseal as ts
from rich.console import Console
from rich.tree import Tree
from rich.table import Table
import warnings
from tqdm import tqdm
from fhenn import _l10n

from fhenn.constants import Constants
from fhenn.nn.cnn2d import CNN2D, EncryptedCNN2D

warnings.simplefilter(action="ignore", category=FutureWarning)

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

app = typer.Typer(name="cnn")


class SupportedDataset(str, Enum):
    """Supported datasets for training and testing."""

    mnist = "mnist"
    """ The MNIST dataset. [URL](https://yann.lecun.com/exdb/mnist/) """

    fashion_mnist = "fashion_mnist"
    """ The Fashion-MNIST dataset. [URL](https://github.com/zalandoresearch/fashion-mnist) """

    emnist = "emnist"
    """ The EMNIST dataset. [URL](https://www.nist.gov/itl/products-and-services/emnist-dataset) """

    kmnist = "kmnist"
    """ The Kuzushiji-MNIST dataset. [URL](http://codh.rois.ac.jp/kmnist/index.html.en) """

    qmnist = "qmnist"
    """ The QMNIST dataset. [URL](https://github.com/facebookresearch/qmnist) """


# Set the seed for reproducibility
torch.manual_seed(21)


@app.callback()
def callback():
    """
    Display a header-like message for all sub-commands.
    """
    typer.echo(_l10n("Convolutional Neural Network (CNN) training and testing."))


def _train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion,
    optimizer,
    device,
    n_epochs: int,
):
    model.train()
    p_bar = tqdm(
        total=n_epochs,
        desc=_l10n("Training"),
        leave=True,
        colour="blue",
        unit=_l10n("epoch"),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]",
    )
    for _ in range(n_epochs):
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

        p_bar.set_description(
            _l10n("Training (loss {train_loss:.6f})").format(train_loss=train_loss)
        )
        p_bar.update()

    p_bar.close()
    # model in evaluation mode
    model.eval()
    return model


@app.command(
    short_help=_l10n("The training command"),
)
def train(
    model_output_path: str = typer.Argument(
        help=_l10n("The path to save the trained model."),
        writable=True,
        exists=False,
        resolve_path=True,
    ),
    batch_size: Optional[int] = typer.Option(
        help=_l10n("The batch size to use for training."), default=64
    ),
    dataset: Optional[SupportedDataset] = typer.Option(
        help=_l10n("The dataset to use for training."), default=SupportedDataset.mnist
    ),
    epochs: Optional[int] = typer.Option(
        help=_l10n("The number of epochs to train the model."), default=10
    ),
):
    console = Console()
    if dataset == SupportedDataset.mnist:
        chosen_dataset = datasets.MNIST
    elif dataset == SupportedDataset.fashion_mnist:
        chosen_dataset = datasets.FashionMNIST
    elif dataset == SupportedDataset.emnist:
        chosen_dataset = datasets.EMNIST
    elif dataset == SupportedDataset.kmnist:
        chosen_dataset = datasets.KMNIST
    elif dataset == SupportedDataset.qmnist:
        chosen_dataset = datasets.QMNIST
    else:
        typer.secho(_l10n("Invalid dataset specified."), bg=typer.colors.RED)
        raise typer.Exit(code=1)

    train_data = (
        chosen_dataset(
            Constants.DATA_DIRECTORY,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        if dataset != SupportedDataset.emnist
        else chosen_dataset(
            Constants.DATA_DIRECTORY,
            split="mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    model = CNN2D()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    config_tree = Tree(_l10n("Configuration"))
    config_tree.add(_l10n("Model: {model}").format(model=model))
    config_tree.add(_l10n("Batch size: {batch_size}").format(batch_size=batch_size))
    config_tree.add(_l10n("Dataset: {dataset}").format(dataset=dataset.value))
    config_tree.add(_l10n("Epochs: {epochs}").format(epochs=epochs))
    config_tree.add(_l10n("Device: {device}").format(device=model.device))
    config_tree.add(_l10n("Criterion: {criterion}").format(criterion=criterion))
    config_tree.add(_l10n("Optimizer: {optimizer}").format(optimizer=optimizer))
    console.print(config_tree)

    model = _train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        device=model.device,
        optimizer=optimizer,
        n_epochs=epochs,
    )
    torch.save(model.state_dict(), model_output_path)
    typer.echo(
        _l10n("Model saved to {model_output_path}").format(
            model_output_path=model_output_path
        )
    )


def _test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion,
    device,
    classes,
):
    console = Console()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for _ in range(len(classes)))
    class_total = list(0.0 for _ in range(len(classes)))

    p_bar = tqdm(
        total=len(test_loader),
        desc=_l10n("Testing in batches"),
        leave=True,
        colour="yellow",
        unit=_l10n("batch"),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]",
    )
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
        p_bar.update()
    p_bar.close()

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader)
    typer.echo(_l10n("Test loss: {test_loss:.6f}\n").format(test_loss=test_loss))

    table = Table()
    table.add_column(_l10n("Label"), justify="center", no_wrap=True)
    table.add_column(_l10n("Accuracy"), justify="center")
    table.add_column(
        _l10n("Observations (correct/total)"), justify="center", no_wrap=True
    )

    for idx, label in enumerate(classes):
        table.add_row(
            f"{label}",
            f"{int(100 * class_correct[idx] / class_total[idx])}%",
            f"{int(np.sum(class_correct[idx]))}/{int(np.sum(class_total[idx]))}",
        )

    table.add_section()

    table.add_row(
        _l10n("Overall"),
        f"{int(100 * np.sum(class_correct) / np.sum(class_total))}%",
        f"{int(np.sum(class_correct))}/{int(np.sum(class_total))}",
        style="bold",
        end_section=True,
    )

    console.print(table)


@app.command(
    short_help=_l10n("The plaintext test command."),
    help=_l10n("This command performs test on a previously trained model."),
)
def test(
    model_input_path: str = typer.Argument(
        help=_l10n("The path to the trained model."),
        readable=True,
        exists=True,
        resolve_path=True,
    ),
    batch_size: Optional[int] = typer.Option(
        help=_l10n("The batch size to use for testing."), default=64
    ),
    dataset: Optional[SupportedDataset] = typer.Option(
        help=_l10n("The dataset to use for testing."),
        default=SupportedDataset.mnist,
    ),
):
    console = Console()
    if dataset == SupportedDataset.mnist:
        chosen_dataset = datasets.MNIST
    elif dataset == SupportedDataset.fashion_mnist:
        chosen_dataset = datasets.FashionMNIST
    elif dataset == SupportedDataset.emnist:
        chosen_dataset = datasets.EMNIST
    elif dataset == SupportedDataset.kmnist:
        chosen_dataset = datasets.KMNIST
    elif dataset == SupportedDataset.qmnist:
        chosen_dataset = datasets.QMNIST
    else:
        typer.secho(_l10n("Invalid dataset specified."), bg=typer.colors.RED)
        raise typer.Exit(code=1)

    test_data = (
        chosen_dataset(
            Constants.DATA_DIRECTORY,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        if dataset != SupportedDataset.emnist
        else chosen_dataset(
            Constants.DATA_DIRECTORY,
            split="mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )
    model = CNN2D()
    model.load_state_dict(torch.load(model_input_path))
    typer.echo(
        _l10n("Loaded model from {model_input_path}").format(
            model_input_path=model_input_path
        )
    )
    criterion = torch.nn.CrossEntropyLoss()

    config_tree = Tree(_l10n("Configuration"))
    config_tree.add(_l10n("Model: {model}").format(model=model))
    config_tree.add(_l10n("Batch size: {batch_size}").format(batch_size=batch_size))
    config_tree.add(_l10n("Dataset: {dataset}").format(dataset=dataset.value))
    config_tree.add(_l10n("Device: {device}").format(device=model.device))
    config_tree.add(_l10n("Criterion: {criterion}").format(criterion=criterion))
    console.print(config_tree)

    _test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=model.device,
        classes=test_data.classes,
    )


def _enc_test(
    context: ts.context,
    encrypted_model,
    test_loader,
    criterion,
    kernel_shape,
    stride,
    classes,
):
    console = Console()
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for _ in range(len(classes)))
    class_total = list(0.0 for _ in range(len(classes)))

    p_bar = tqdm(
        total=len(test_loader),
        desc=_l10n("Testing encrypted input"),
        leave=True,
        colour="yellow",
        unit=_l10n("classification"),
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}{postfix}]",
    )

    for data, target in test_loader:
        # Encoding and encryption
        x_enc, windows_nb = ts.im2col_encoding(
            context,
            data.view(data.shape[-2], data.shape[-1]).tolist(),
            kernel_shape[0],
            kernel_shape[1],
            stride,
        )
        # Encrypted evaluation
        encrypted_output = encrypted_model(x_enc, windows_nb)
        # Decryption of result
        output = encrypted_output.decrypt()
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
        p_bar.update()

    p_bar.close()
    # calculate and print avg test loss
    test_loss = test_loss / sum(class_total)
    typer.echo(_l10n("Test loss: {test_loss:.6f}\n").format(test_loss=test_loss))

    table = Table()
    table.add_column(_l10n("Label"), justify="center", no_wrap=True)
    table.add_column(_l10n("Accuracy"), justify="center")
    table.add_column(
        _l10n("Observations (correct/total)"), justify="center", no_wrap=True
    )

    for idx, label in enumerate(classes):
        table.add_row(
            f"{label}",
            f"{int(100 * class_correct[idx] / class_total[idx])}%",
            f"{int(np.sum(class_correct[idx]))}/{int(np.sum(class_total[idx]))}",
        )

    table.add_section()

    table.add_row(
        _l10n("Overall"),
        f"{int(100 * np.sum(class_correct) / np.sum(class_total))}%",
        f"{int(np.sum(class_correct))}/{int(np.sum(class_total))}",
        style="bold",
        end_section=True,
    )

    console.print(table)


@app.command(
    short_help=_l10n("The encrypted test command."),
    help=_l10n("This command performs encrypted test on a previously trained model."),
)
def encrypted_test(
    model_input_path: str = typer.Argument(
        help=_l10n("The path to the trained model."), readable=True, exists=True
    ),
    dataset: Optional[SupportedDataset] = typer.Option(
        help=_l10n("The dataset to use for testing."),
        default=SupportedDataset.mnist,
    ),
):
    console = Console()

    if dataset == SupportedDataset.mnist:
        chosen_dataset = datasets.MNIST
    elif dataset == SupportedDataset.fashion_mnist:
        chosen_dataset = datasets.FashionMNIST
    elif dataset == SupportedDataset.emnist:
        chosen_dataset = datasets.EMNIST
    elif dataset == SupportedDataset.kmnist:
        chosen_dataset = datasets.KMNIST
    elif dataset == SupportedDataset.qmnist:
        chosen_dataset = datasets.QMNIST
    else:
        typer.secho("Invalid dataset specified.", bg=typer.colors.RED)
        raise typer.Exit(code=1)

    test_data = (
        chosen_dataset(
            Constants.DATA_DIRECTORY,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        if dataset != SupportedDataset.emnist
        else chosen_dataset(
            Constants.DATA_DIRECTORY,
            split="mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    )
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True
    )
    model = CNN2D()
    if str(model.device) != "cpu":
        typer.secho(
            _l10n(
                "Even if the model is on a {device} device, "
                "the encrypted test is done on CPU because encrypted "
                "queries are not supported on GPU-like devices."
            ).format(device=model.device),
            fg=typer.colors.BRIGHT_RED,
            bg=typer.colors.BRIGHT_YELLOW,
        )
    model.load_state_dict(torch.load(model_input_path))
    typer.echo(
        _l10n("Loaded model from {model_input_path}").format(
            model_input_path=model_input_path
        )
    )
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

    config_tree = Tree(_l10n("Configuration"))
    config_tree.add(_l10n("Model: {model}").format(model=model))
    config_tree.add(_l10n("Batch size: {batch_size}").format(batch_size=batch_size))
    config_tree.add(_l10n("Dataset: {dataset}").format(dataset=dataset.value))
    config_tree.add(_l10n("Criterion: {criterion}").format(criterion=criterion))
    config_tree.add(
        _l10n("Kernel shape: {kernel_shape}").format(kernel_shape=kernel_shape)
    )
    config_tree.add(_l10n("Stride: {stride}").format(stride=stride))
    config_tree.add(
        _l10n("FHE scheme: {fhe_scheme}").format(fhe_scheme=ts.SCHEME_TYPE.CKKS)
    )
    console.print(config_tree)

    _enc_test(
        context=context,
        encrypted_model=EncryptedCNN2D(model),
        test_loader=test_loader,
        criterion=criterion,
        kernel_shape=kernel_shape,
        stride=stride,
        classes=test_data.classes,
    )
