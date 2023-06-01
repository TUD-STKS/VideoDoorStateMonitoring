# "Non-Standard Echo State Networks for Video Door State Monitoring"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv venv

.\venv\Scripts\activate.ps1

python.exe -m pip install -r requirements.txt
python.exe -m pip install --editable .
jupyter-lab.exe

deactivate
