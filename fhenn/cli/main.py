import typer
from fhenn.cli import version
from fhenn.cli import cnn

app = typer.Typer(
    name="FHENN",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=True,
    # The following is for security reasons.
    pretty_exceptions_show_locals=False,
)

app.add_typer(version.app)
app.add_typer(cnn.app)

if __name__ == "__main__":
    app()
