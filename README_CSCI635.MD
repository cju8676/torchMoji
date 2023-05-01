# CSCI 635 Final Project - Group 9

## Original README
For an overview on the original project structure, backend, tests, data, and disclaimers, see the original README.MD

## Setup on CS Machine README
For an overview on how to set up and run the project on a CS machine, please see [CSServerSetup.MD](CSServerSetup.MD)

It is preferable to run on one of the GPU compute nodes for the best performance/training time

## Overview
- Framework: PyTorch
- GPU: Increases performance considerably, but code will just use CPU if GPU is unavailable
- Setup on CS server uses a Miniconda environment with utilizing [torchmoji_env.yml](torchmoji_env.yml) YML file to create and run a suitable environment for the code
- As an alternative, the [setup.py](setup.py) file could be run to create the env using 
```bash
pip install -e .
```
- Ensure you have the original model in model/ directory, if not, it can be downloaded [here](https://www.dropbox.com/s/q8lax9ary32c7t9/pytorch_model.bin?dl=0) as mentioned in the original README

## Group 9's Changes
The majority of code is left as is, but files and locations where our code is changed to benefit our new experiment will contain file headers and comments that link to the specific lines that were modified.

Specifically, look out for file headers containing:
```
    CS635 Final Project - Group 9

    Modifications: ...
```

Search for these specific code comments:
```
########################
# -- Modified -- CS635 #
########################
```
Note: small fixes to get the code to run are not headered by a --- Modified --- banner, only significant
experiment changes...
For full changes please see [GitHub](https://github.com/huggingface/torchMoji/compare/master...cju8676:torchMoji:master)

## Running Code
As mentioned, please see CSServerSetup.MD for a full installation and run through of running code.
Quickly, you can run unit tests with a configured environment here:
```
conda activate torchmoji_env
cd tests
nosetests --nocapture [TESTFILE]:[TEST_FUNCTION]
```
or you can run all tests by doing:
```
nosetests -v --nocapture
```
For example:
```
nosetests --nocapture test_finetuning.py:test_finetune_chain_thaw_10
```
will run the ```test_finetune_chain_thaw_10``` function only.

To run our Experiment for the CS635 final project:
```
conda activate torchmoji_env
cd scripts

/home/stu1/s15/[YOUR_RIT_ACCOUNT]/miniconda3/envs/torchmoji_env/bin/python cs635experiment.py [LAYER_COUNT] > [OUTPUT_TEXT]
```
this command utilizes the miniconda environment installed to *YOUR_RIT_ACCOUNT* to run our experiment script with *LAYER_COUNT* argument being **128, 256, 512 (original/default), or 1024** and pipe standard out to *OUTPUT_TEXT* file to be viewed later