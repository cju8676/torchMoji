# Deepmoji Setup Steps for RIT CS machine

1. choose a CS node (i chose weasley.cs.rit.edu) and SSH into it

- https://wiki.cs.rit.edu/index.php/ClientNodes

2. download miniconda for your account

    - first download linux script from here https://docs.conda.io/en/latest/miniconda.html#linux-installers

    - SFTP the script into your CS account and with the SSH connection run:
```
	$ bash Miniconda3-py310_23.1.0-1-Linux-x86_64
```

- accept the license and let it install into your dir

3. Grab DeepMoji source code either from project submission or my GitHub
```
    git clone https://github.com/cju8676/torchMoji.git
```
4. create a conda environment for your code to run in

    - Verify you are in the root directory where ```torchmoji_env.yml``` is located

    - then run the following:
```bash
	$ . ~/.bashrc
	$ conda env create -f torchmoji_env.yml
	$ conda activate torchmoji_env
```
5. fix a small issue in one of the tests

- goto torchMoji/torchmoji/lstm.py line 78 
	and change 
    ```input, batch_sizes = input```
    to
	```input, batch_sizes, _, _ = input```
6. run tests and confirm they all pass
```
	$ cd tests
	$ nosetests -v -a '!slow'
```

7. Run our added experiment code:
```
conda activate torchmoji_env
cd [location_of_code]/scripts
    
/home/stu1/s15/[YOUR_RIT_ACCOUNT]/miniconda3/envs/torchmoji_env/bin/python cs635experiment.py [LAYER_COUNT] > [OUTPUT_TEXT]
```
this command utilizes the miniconda environment installed to *YOUR_RIT_ACCOUNT* to run our experiment script with *LAYER_COUNT* argument being **128, 256, 512 (original/default), or 1024** and pipe standard out to *OUTPUT_TEXT* file to be viewed later