#!/bin/bash
# This script is used to initialize the messages.po files. Run this only if the messages.po files are not present.

# Language: English
msginit --no-translator --no-wrap -l en_US.UTF8 -o fhenn/locales/en/LC_MESSAGES/messages.po -i fhenn/locales/messages.pot
# Language: Japanese
msginit --no-translator --no-wrap -l ja_JP.UTF8 -o fhenn/locales/ja/LC_MESSAGES/messages.po -i fhenn/locales/messages.pot
