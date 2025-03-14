# boilerplate
 A boilerplate for typical ML projects.

    ✅ Configurable experiments that make it extremely easy (and traceable) to run different experiments with different models or different datasets.
    ✅ Run logs that keep a record of every experiment with the configuration file, arguments and training logs.
    ✅ Enable logging to tensorboard or WandB through the config file.
    ✅ Early stopping, learning rate scheduling, storing and resuming from checkpoints, and all the good stuff.
    ✅ It also has demo notebooks that make it easier for you to showcase qualitative results without duplicating code and causing inconsistency.

See the detailed [features list](#features-list).

### Table of Contents
**[Cloning as a GitHub template](#cloning-as-a-github-template)**<br>
**[Package Structure](#package-structure)**<br>
- **[Configs Module](#configs-module-configs)**<br>
- **[Datasets Module](#datasets-module-datasets)**<br>
- **[Models Module](#models-module-models)**<br>
- **[Utils Module](#utils-module-utils)**<br>
- **[Demo Directory](#demo-directory-demo)**<br>

**[Adding a new Model class](#adding-a-new-model-class)**<br>
**[Adding a new Dataset class](#adding-a-new-dataset-class)**<br>
**[Other instructions](#other-instructions)**<br>
**[Current Features List](#current-features-list)**<br>
**[Features to be Added](#features-to-be-added)**<br>

- - - -

### Cloning as a GitHub template
To start a new repository from this boilerplate, click the green "Use this template" button at the top of the main repo page. This is not a fork, but the new repository will be marked that it was started from this GitHub template. Alternatively, you can download this repository and use in your ML project.

After the repo is ready to be used, the following command must be run in the repo's root to install it as Python package in your Python environment.
```
pip install -e .
```
THe argument `-e` is for an editable install.

### Package Structure

    boilerplate/
    ├── train.bat
    ├── train.sh
    ├── train.py
    ├── pyproject.toml
    ├── ...
    ├── configs/
    │   ├── __init__.py
    │   ├── _defaults.py
    │   ├── ...
    │   └──
    ├── datasets/
    │   ├── __init__.py
    │   ├── ...
    │   └──
    ├── models/
    │   ├── __init__.py
    │   ├── ...
    │   └──
    ├── utils/
    │   ├── __init__.py
    │   ├── ...
    │   └── utils.py
    └── demo/
        ├── demo.ipynb
        ├── test.py
        ├── test.sh
        ├── test.bat
        └── ...

##### Configs Module ([/configs](/configs))
The configs module can be imported using `import bp_configs`, it contains the config files that carry the definition of your runs. It uses the YACS configuration system (See [github.com/rbgirshick/yacs](https://github.com/rbgirshick/yacs)).

##### Datasets Module ([/datasets](/datasets))
The datasets module can be imported using `import bp_datasets`, it contains all the different dataset classes that will be used in the repository. The main export out of this module is the `DATASETS` dictionary. See the [adding a new dataset](/README.md#adding-a-new-dataset-class) instructions.

##### Models Module ([/models](/models))
The models module can be imported using `import bp_models`, it contains all the different model classes that will be used in the repository. The main export out of this module is the `MODELS` dictionary. See the [adding a new model](/README.md#adding-a-new-model-class) instructions.

##### Utils Module ([/utils](/utils))
The utils module can be imported using `import bp_utils`, it can be used to keep the definitions of functions that are frequently used across the repository.

##### Demo Directory ([/demo](/demo))
The is a directory that can be used for scripts that test and demonstrate the models that you train. It contains an example Jupyter Notebook that show how to correctly import model and dataset classes, as well as some boilerplate code for basic generic tasks.

The `demo` directory includes a test script for inference in the `test.py`, as well as an example BASH file and Batch file for Linux and Windows system respectively.
You can run `python demo/test.py --help` to get details about the required inline arguments. This test script run using a Run Name, a checkpoint, or some specific configuration file.


### Adding a new Model class

1. Add the model class PY file/files to the [/models](/models) directory (See example [acme_model.py](/models/acme_model.py)).
2. Modify the [`__init__.py`](/models/__init__.py) file in the [/models](/models) directory, and add the following.
```python
# import the model class
from .new_model import NewModel

# this dict definition already exists
# required once
MODELS = {}

# add a function that returns the new model
# this function should take a single `config` argument
# this helps in having a uniform interface to all included models
# no matter what arguments the model class constructor requires
def get_NewModel(cfg):
    return NewModel(
        input_size=cfg.MODEL.INPUT_SIZE,
        num_classes=cfg.MODEL.NUM_CLASSES,
    )

# add the new function to the MODELS dictionary
MODELS["NewModel"] = get_NewModel

# repeat for any other models
#
```
3. Document your model in the models [README](/models/README.md).

### Adding a new Dataset class

1. Add the dataset class PY file/files to the [/datasets](/datasets) directory (See example [acme_dataset.py](/datasets/acme_dataset.py)).
2. Modify the [`__init__.py`](/datasets/__init__.py) file in the [/datasets](/datasets) directory, and add the following.
```python
# import the dataset class
from .new_dataset import NewDataset

# this dict definition already exists
# required once
DATASETS = {}

# add a function that returns the new model
# this function can take the shown 4 arguments
# this helps in having a uniform interface to all included datasets
# no matter what arguments the dataset class constructor requires
def get_NewDataset(data_root, split, cfg=None, logger=None):
    return NewDataset(
        data_root=data_root,
        split=split,
    )

# add the new function to the MODELS dictionary
DATASETS["NewDataset"] = get_NewDataset

# repeat for any other datasets
#
```
3. Document your dataset in the datasets [README](/datasets/README.md).

### Run Logs
Every time the `train.py` file is run, a run folder is created (in `/runs` by default) and named with the same name as the run. This run directory is used to store logs and information about the run.

```
    boilerplate/
    └── runs/
        ├── __init__.py
        ├── checkpoints
        │   ├ ...
        │   └ chkpt_best.pt
        ├── log.txt
        ├── config.yml
        └── arguments.yml
```

Each run folder has the complete run configuration in the `config.yml` file, the given inline arguments in the `arguments.yml` file, the console logs in the `log.txt` file, and the stored checkpoints in the `checkpoints` directory.

### Current Features List
- Complete training and evaluation loop.
- Saving a checkpoint at a specified epochs interval.
- Saving the best checkpoint based on the evaluation loss.
- Resuming from a checkpoint specified through inline arguments.
- Running Evaluation at a specified epochs interval.
- Logging training loss and evaluation loss on Tensorboard and/or wandb.
- Tracking multiple different runs and storing the runs outputs.
- Storing run logs that include a summary of the model, the provided inline arguments for the run, the checkpoints, and the Tensorboard logs.
- Early stopping (`--patience` argument).
- Supports configurable multiple models.
- Supports configurable multiple datasets.
- Supports configurable multiple optimizers.
- Supports configurable multiple loss functions.
- Supports learning rate schedulers.
- Uses config files for variables that describe the model, dataset, and training procedure.
- Uses inline arguments for variables that are not recommended to be commited like data paths.

### Features to be Added
- Better and more comprehensive logging.
- TBD
