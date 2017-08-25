## Description of code files:
 - `main.py` -- The main file to run for training and testing.
 - `model.py` -- defines the whole generative model
 - `inputs.py` -- defines all the command line argument to `main.py` file
 - `ops.py` -- defines all the utility functions
 - `evaluation.py` -- define all the evaluation functions

## How to run:
To train and test the model, run
```
$ cd src/
$ python main.py --dataset='../data/bibtex.mat'
```
The default value for other command line arguments are defined in `inputs.py`. These arguments can easily be changed from command line run.

## Bash Scripts
The bash scripts(`*.sh` files) are included in the `script/` folder. These files contains the loosly tuned parameters for some of the datasets.