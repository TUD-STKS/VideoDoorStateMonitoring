# Non-Standard Echo State Networks for Video Door State Monitoring
## Metadata
- Author: [Peter Steiner](mailto:peter.steiner@tu-dresden.de),
- [Azarakhsh Jalalvand](mailto:azarakhsh.jalalvand@princeton.edu), 
  [Peter Birkholz](mailto:peter.birkholz@tu-dresden.de)
- Conference: International Joint Conference ont Neural Networks (IJCNN ) 2023,
  Queensland, Australia
- Weblink:
[https://github.com/TUD-STKS/VideoDoorStateMonitoring](https://github.com/TUD-STKS/VideoDoorStateMonitoring)

## Summary and Contents
This is the repository containing the code to reproduce the content of the research paper
"Non-Standard Echo State Networks for Video Door State Monitoring". It contains a Jupyter 
notebook, pre-trained models and the code to obtain all Figures and results in the paper.

We propose to use the following structure of this README:
- File list:
    - The file list contains all files provided in the repository together with a 
    short description.
- Usage:
    - How can users get started with your research code. This contains setting up a 
    installing packages in a virtual environment `venv` and running one `main.py` that
    includes your main code. 
    - Very important and often forgotten: How can the data to reproduce the results be
    obtained?
    - In case of a Jupyter notebook, it is strongly recommended to add a link to 
    [Binder](https://mybinder.org/).
- Referencing:
    - How can your work be cited? Ideally, provide a bibtex entry of the paper.

## File list
- The following scripts are provided in this repository
    - `scripts/run.sh`: UNIX Bash script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.sh`: UNIX Bash script to start the Jupyter Notebook for 
   the paper.
    - `scripts/run.bat`: Windows batch script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.bat`: Windows batch script to start the Jupyter Notebook 
  for the paper.
- The following python scripts and modules are provided in `src`
    - `src/dataset`: Module for dataset handling.
    - `src/input_to_node`: Custom InputToNode building blocks for the KM-ESN and 
      simple ESN architectures.
    - `src/node_to_node`: Custom NodeToNode building blocks for the simple ESN 
      architectures.
    - `src/main.py`: The main script to train all models.
    - `Video_Door_State_Monitoring.ipynb`: The Jupyter notebook to reproduce all results.
- `requirements.txt`: Text file containing all required Python modules to be installed. 
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code. You can find 
- `data/`: The directory `data` is prepared to hold the dataset. Please request the 
  dataset from the authors.
- `results/`: The directory `results` is prepared to hold the trained models. Please 
  request the pre-trained models from the authors.
- `.gitignore`: Command file for Github to ignore specific files.

## Usage
The easiest way to reproduce the results is to run the Notebook online using Binder.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TUD-STKS/VideoDoorStateMonitoring/HEAD)

To run the scripts or to start the Jupyter Notebook locally, at first, please ensure 
that you have a valid Python distribution installed on your system. Here, at least 
Python 3.8 is required.

You can then call `run_jupyter-lab.ps1` or `run_jupyter-lab.sh`. This will install a new 
[Python venv](https://docs.python.org/3/library/venv.html), which is our recommended way 
of getting started.

The pre-trained models that are part of this repository can be re-trained using the 
script `main.py`. The easiest way to reproduce the results is to either download and
extract this Github repository in the  desired directory, open a Linux Shell and call 
`run.sh` or open a Windows PowerShell and call `run.ps1`. 

In that way, again, a [Python venv](https://docs.python.org/3/library/venv.html) is 
created, where all required packages (specified by `requirements.txt`) are installed.
Afterwards, the script `main.py` is excecuted with all default arguments activated in
order to reproduce all results in the paper.

If you want to suppress any options, simply remove the particular option.


## License and Referencing
This program is licensed under the BSD 3-Clause License. If you in any way use this
code for research that results in publications, please cite our original
article

```
@inproceedings{steiner2023non-standard,
  author    = {Steiner, Peter and Jalalvand, Azarakhsh and Birkholz, Peter},
  booktitle = {2023 International Joint Conference on Neural Networks (IJCNN)},
  title     = {Non-Standard Echo State Networks for Video Door State Monitoring},
  year      = {2023},
  doi       = {...},
  pages     = {1--8},
}
```
## Appendix
For any questions, do not hesitate to open an issue or to drop a line to
[Peter Steiner](mailto:peter.steiner@tu-dresden.de)
