# Extract translatable texts
xgettext \
    --no-wrap \
    --package-name=FHENN \
    --keyword=_l10n \
    -d messages \
    -o fhenn/locales/messages.pot \
    --from-code UTF-8 \
    fhenn/cli/*.py
