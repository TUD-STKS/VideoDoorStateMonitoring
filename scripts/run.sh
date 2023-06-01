#!/bin/sh

# "Non-Standard Echo State Networks for Video Door State Monitoring"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install --editable .
python3 src/main.py --fit_basic_esn

deactivate
