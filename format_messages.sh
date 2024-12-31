#!/bin/bash
# This script is used to format the messages.po files into messages.mo files
# Language: English
msgfmt -o fhenn/locales/en/LC_MESSAGES/messages.mo fhenn/locales/en/LC_MESSAGES/messages.po
# Language: Japanese
msgfmt -o fhenn/locales/ja/LC_MESSAGES/messages.mo fhenn/locales/ja/LC_MESSAGES/messages.po
# Add other languages, when necessary
