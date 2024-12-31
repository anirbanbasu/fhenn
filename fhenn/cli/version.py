from datetime import datetime
from importlib import metadata as importlib_metadata

from typing import Optional
import typer

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from fhenn.constants import Constants
from fhenn import _l10n

# # Basic localisation for this version.py.
# # See: https://lokalise.com/blog/translating-apps-with-gettext-comprehensive-tutorial/
# # See: https://phrase.com/blog/posts/translate-python-gnu-gettext/
# # See: https://docs.python.org/3/library/gettext.html

# import locale
# import gettext

# gettext.bindtextdomain("messages", "fhenn/locales")
# gettext.textdomain("messages")

# locale.setlocale(locale.LC_CTYPE, "ja_JP.UTF-8")

# locale = locale.getlocale(category=locale.LC_CTYPE)

# lang = gettext.translation("messages", localedir="fhenn/locales", languages=[locale[0]])
# lang.install()
# _ = lang.gettext
# ngettext = lang.ngettext

app = typer.Typer()


@app.command(
    name="version",
    help=_l10n(
        "Show the version and optionally, the extended metadata of the {app_name} package."
    ).format(app_name=Constants.APP_NAME),
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
            title=_l10n("{app_name} Metadata as of {timestamp}").format(
                app_name=Constants.APP_NAME,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
            title_justify="right",
            title_style="bold",
            safe_box=True,
        )
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        md_in_desc = False
        for k, v in importlib_metadata.metadata(Constants.PACKAGE_NAME).json.items():
            if k == "description_content_type" and v == "text/markdown":
                md_in_desc = True
                continue
            if k == "description":
                if md_in_desc:
                    table.add_row(
                        _l10n("{k}{crlf}(picked up from README)").format(
                            k=k, crlf=Constants.CRLF
                        ),
                        Markdown(v),
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
        typer.echo(
            _l10n("{pkg_name} {metadata_version}").format(
                pkg_name=Constants.APP_NAME,
                metadata_version=importlib_metadata.version(Constants.PACKAGE_NAME),
            )
        )
