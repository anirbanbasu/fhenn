# Basic localisation for this version.py.
# See: https://lokalise.com/blog/translating-apps-with-gettext-comprehensive-tutorial/
# See: https://phrase.com/blog/posts/translate-python-gnu-gettext/
# See: https://docs.python.org/3/library/gettext.html

import locale
import gettext

DEFAULT_LANGUAGE = "en_US"
LOCALE_DIR = "fhenn/locales"
DOMAIN = "messages"

try:
    locale = locale.getlocale(category=locale.LC_CTYPE)
    lang = gettext.translation(
        domain=DOMAIN, localedir=LOCALE_DIR, languages=[locale[0]]
    )
except FileNotFoundError:
    # Assume that the default language is available
    lang = gettext.translation(
        domain=DOMAIN, localedir=LOCALE_DIR, languages=[DEFAULT_LANGUAGE]
    )

lang.install()
# Localisation installed here for global use
_l10n = lang.gettext

# TODO: Pluralisation support not yet implemented
ngettext = lang.ngettext
