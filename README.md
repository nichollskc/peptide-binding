# Peptide binding prediction

[![Build Status](https://travis-ci.org/nichollskc/peptide-binding.svg?branch=master)](https://travis-ci.org/nichollskc/peptide-binding) [![Coverage Status](https://coveralls.io/repos/github/nichollskc/peptide-binding/badge.svg?branch=master)](https://coveralls.io/github/nichollskc/peptide-binding?branch=master)

The aim of this project is to be able to train a statistical model to predict whether two peptides will bind together. There are two parts of the project:

1. Pipeline to generate datasets
2. Models that can be trained using these datasets

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project uses [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to organise the required packages. Check if conda is already installed by checking the output of `which conda`:

```
$ which conda
~/miniconda3/condabin/conda
```

Full instructions are available from [conda's own documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) but for most linux users the following instructions are sufficient:

```
# Download the install script
sudo apt-get update
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
# Install miniconda in the user's home directory
bash miniconda.sh -b -p $HOME/miniconda
# Add the install directory to the path
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

### Installing

This section describes how to download and set up the repository. The steps can also be found in the `.travis.yml` file which is used for test builds.

First clone the github repository and move into the created folder:

```
git clone https://github.com/nichollskc/peptide-binding.git
cd peptide-binding
```

Create the conda environment named peptidebinding and activate it:

```
conda env create -f environment.yml -n peptidebinding
conda activate peptidebinding
```

Install extra packages that cannot be installed with conda. It is [recommended](https://www.anaconda.com/using-pip-in-a-conda-environment/) to perform any pip install steps after creating the conda environment.

```
pip install sacred
```

Initialise the cd-hit repository which is a submodule of this project:

```
git submodule update --init --recursive
cd seq-align && make
cd -
```

#### Directory structure

```
peptide-binding
# Input files
├── IDs
├── cleanPDBs2
├── icMatrix
# Scripts and tests
├── peptidebinding
├── tests
├── seq-align
# Generating plots and exploring data
├── report
# Processed files as part of pipeline and logs
├── processed
├── logs
# Final product - datasets for training
├── datasets
# Trained models and metrics
└── models
```

#### Adding data

The pipeline relies on three types of input file for each protein used from the Protein Data Bank:

* [PDB files](cleanPDBs2), processed versions of files available from the [Protein Data Bank](http://www.rcsb.org/)
* [IDs files](IDs), space separated files giving the chain, residue number and residue type of each residue in the protein
* [matrix files](icMatrix), binary matrix files describing both which fragments are CDR-like and also which pairs of residues interact (though only information about CDR-like fragments is used)

A small number of sets of these files are provided with the repository for testing. Initial work has used roughly 40,000 sets of these files.

### Running pipeline

Snakemake is used to organise the pipeline. It is a python-like version of `make` and is well-documented in the [snakemake documentation](https://snakemake.readthedocs.io/en/stable/).

The basic command is:

```
snakemake --use-conda <target>
```

Snakemake will then evaluate the steps required to create the target file from the existing files, and run each of these steps in turn.

`<target>` can either be a file to be generated, or a rule e.g. `test` or `all`. If no target is given, the first rule in the file will be taken as the target.

#### Useful snakemake commands

###### Print commands
It is often useful to run with the flag `--printshellcmds` for debugging purposes. This prints out the shell commands once all the arguments have been inserted.

###### Submit jobs to cluster
Snakemake can submit individual jobs to a cluster system. For example, to submit using qsub use the following command:

```
snakemake --cluster "qsub -q sl -V" --printshellcmds --use-conda --jobs 100 <target>
```

The `--jobs 100` flag tells snakemake the maximum number of jobs it can have submitted to the cluster queue at any one time.

NOTE: If the repository is on the marcopolo computing cluster in the /nodescratch space, follow the rules [here](#marcopolo-cluster-notes).

###### Test rule
There is a rule in the pipeline that can be used for testing. As long as there are few sets of input files (PDBs/icMat/IDs), all the jobs in the test pipeline will take 5-10 minutes. WARNING: even the test rule will take a long time if there are many PDB files.

```
snakemake --printshellcmds --use-conda test
```

###### Dry-run

The flag `--dryrun` tells snakemake just to calculate the required steps, but not actually perform them.

```
snakemake --use-conda --dryrun <target>
```

### Using datasets

Most datasets generated have a directory structure such as the one below.

```
datasets/beta/small/10000/clust/
├── test
│   ├── data_fingerprints.npz
│   └── labels.npy
├── training
│   ├── data_fingerprints.npz
│   └── labels.npy
└── validation
    ├── data_fingerprints.npz
    └── labels.npy
```

The file format for the molecular fingerprints representation is the sparse matrix format supplied by the scipy.sparse library. It can be read as follows:

```
import scipy.sparse

X_train = scipy.sparse.load_npz(f"datasets/beta/small/10000/clust/training/data_fingerprints.npz").toarray()
```

### Models

There are a number of model scripts available in the [training folder](peptidebinding/training). These are python scripts which are wrapped using the program [sacred](https://sacred.readthedocs.io/en/latest/quickstart.html). Sacred saves the accuracy and other metrics from each run in a MongoDB database, along with the parameters used for each run. This allows easy comparison of the different models on different datasets and with different parameters.

Models:

* Random forest (runs random search of parameter sets)
* Random forest single (uses a single set of parameters) random_forest_single
* Logistic regression (runs random search of parameter sets) logistic_regression
* Neural network (uses a single set of parameters) neural_network

#### Sacred setup

The model scripts require an account on the MongoDB database `Moorhen`, whose administrator is nichollskc (kcn25). The following environment variables need to be set. They can be set in the user's ~/.bashrc file.

```
export MOORHEN_USERNAME='username'
export MOORHEN_PASSWORD='password'
```

#### Training

To train a predictive model using a dataset with molecular fingerprints representation use a command like the following:

```
python3 -m peptidebinding.training.logistic_regression with representation='fingerprints' dataset='beta/small/10000/clust' num_folds=10 num_param_sets=10
```

This command can be placed in a script so it can be submitted as a cluster job. An example script is [peptidebinding/training/submit_train.sh]. Note that the conda environment must be activated before running the commands. See also special notes about working with the [Marcopolo cluster](#marcopolo-cluster-notes).

NOTE: tensorflow doesn't have the right pre-requisites on marcopolo so the neural_network model cannot be run there.

The model logistic_regression can be replaced with any of random_forest, random_forest_single, logistic_regression or neural_network.

The dataset parameter gives a subdirectory of the `datasets` directory which should contain the following files:

```
datasets/beta/small/10000/clust/
├── test
│   ├── data_fingerprints.npz
│   └── labels.npy
├── training
│   ├── data_fingerprints.npz
│   └── labels.npy
└── validation
    ├── data_fingerprints.npz
    └── labels.npy
```

The test set is not used for fitting the model, but rather is held back for evaluation. The training set is used to fit each individual model, and the validation set is used to report accuracy of the model and to choose between different parameter sets when necessary.

###### Adjusting parameters

In each model script (e.g. peptidebinding/training/neural_network.py) there is a config section which describes the parameters that can be tweaked from the command line. This is also the list of parameters that will be tracked in the database of model metrics.

Here is the config section from the neural network script:

```
@ex.config  # Configuration is defined through local variables.
def cfg():
    """Config definitions for sacred"""
    representation = "bag_of_words"
    dataset = "beta/clust"
    learning_rate = 1e-3
    dropout = 0.3
    units_layer1 = 15
    units_layer2 = 10
    seed = 1342
    epochs = 500
    mb_size = 100
    regularisation_weight = 0.1
```

Any of these parameters can be changed from the command line by adding e.g. `'dropout=0.3'` to the `with` statement:

```
python3 -m peptidebinding.training.neural_network with representation='fingerprints' dataset='beta/small/10000/clust' 'dropout=0.3' 'epochs=100'
```

#### Results database

There is a python module pymongo which can be used to access the results database programmatically. Examples are in the report/query_mongo.py script.

[Omniboard](https://github.com/vivekratnavel/omniboard) is a useful tool to visualise the results. It allows you to filter the models e.g. to only certain datasets and to sort the models by collected metrics. It also allows easy access to the plots and metrics files saved by the model through sacred.

After installation (following instructions [here](https://github.com/vivekratnavel/omniboard/blob/master/docs/quick-start.md)), run the following command to host omniboard locally (replacing USERNAME and PASSWORD):

```
omniboard --mu "mongodb+srv://USERNAME:PASSWORD@moorhen-5migi.mongodb.net/MY_DB?retryWrites=true&w=majority"
```

Then point your browser to [http://localhost:9000/](http://localhost:9000/) to view.

### Further notes

#### Marcopolo cluster notes

If the repository on marcopolo is not in /nodescratch, the ordinary instructions [above](#submit-jobs-to-cluster) can be used.

##### Nodescratch

If the repository on marcopolo is in /nodescratch, special care is needed to ensure that any submitted jobs end up on the node where the repository is located. The flag `-l nodes=node001` can be added to any qsub command to ensure the job is submitted to that node.

To run snakemake using the cluster when the repository is in /nodescratch, the snakemake command itself must be submitted to the right node, and must include a flag to submit individual snakemake jobs to the right node. The script peptidebinding/helper/submit_snakemake.sh can be used for this.

Edit the submit_snakemake.sh script to ensure it references the right repository and the right node (e.g. if the repository is /nodescratch/node001/kcn25/peptide-binding then use nodes=node001, and edit the script so it changes into the peptide-binding directory). Then submit it using:

```
qsub -l nodes=node001 peptidebinding/helper/submit_snakemake.sh
```

To train a predictive model using the cluster, use the `-l nodes=node001` flag to ensure the cluster job will have access to the datasets needed.

```
qsub -l nodes=node001 peptidebinding/training/submit_train.sh
```
