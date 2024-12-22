from datetime import datetime
from importlib import metadata as importlib_metadata
from typing import Optional
import typer

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from fhenn.constants import Constants

app = typer.Typer()


@app.command(
    name="version",
    help="Show the version and optionally, the extended metadata of the FHENN package.",
)
def version(
    more_metadata: Optional[bool] = typer.Option(
        default=False,
        help="Show extended metadata.",
    ),
):
    if more_metadata:
        console = Console()
        table = Table(
            title=f"FHENN Metadata as of {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",
            title_justify="right",
            title_style="bold",
            safe_box=True,
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        md_in_desc = False
        for k, v in importlib_metadata.metadata("fhenn").json.items():
            if k == "description_content_type" and v == "text/markdown":
                md_in_desc = True
                continue
            if k == "description":
                if md_in_desc:
                    table.add_row(
                        f"{k}{Constants.CRLF}(picked up from README)", Markdown(v)
                    )
                else:
                    table.add_row(k, str(v))
            elif k == "classifier" or k == "requires_dist":
                table.add_row(k, Constants.CRLF.join(v))
            else:
                table.add_row(k, str(v))
            table.add_section()
        console.print(table)
    else:
        typer.echo(f"fhenn {importlib_metadata.version('fhenn')}")
