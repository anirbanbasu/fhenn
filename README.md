# Fully Homomorphic Encryption (FHE) and neural networks

This repository contains haphazard experiments with applying fully homomorphic encryption (FHE) on neural networks.

To begin, you must have `poetry` installed. See [installation instructions](https://python-poetry.org/docs/#installation).

You will need [`gettext`](https://www.gnu.org/software/gettext/) on your system to support localisation. Once installed, assuming that you have `msgfmt` on your system, run `./format_messages.sh` to convert the PO files for localisation to their MO equivalents.

Then, run `poetry install` in the directory where you cloned this repository. This will automatically create a virtual environment. Switch to it by invoking `poetry shell`. Then, run `fhenn --help` to see the rather self-explanatory commands available. Alternatively, without switching to the Poetry shell, you can run `poetry run fhenn --help` for the same result.

Note that during the course of running the commands, e.g., training, datasets will be downloaded in the `data` directory relative to the repository root.
