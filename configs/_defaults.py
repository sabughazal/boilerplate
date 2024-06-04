from yacs.config import CfgNode as CN
_C = CN()

"""
Configuration defaults using the YACS framework.
The default values are the currently assigned. Those would be overridden
by the values assigned in the .yml configuartion file.
"""




_C.MODEL = CN()

# (required)
# Model name from the models listed in models/__init__.py.
# New models can be added by adding the model class in a .py file
# to the models folder and then listing it in the models/__init__.py file.
#
_C.MODEL.NAME = "CNN"

# (required)
#
_C.MODEL.INPUT_SIZE = 0

# (required)
# Num of classes
#
_C.MODEL.NUM_CLASSES = 0




_C.DATASET = CN()

# (required)
# Dataset name from the datasets listed in datasets/__init__.py.
# New dataset classes can be added by adding the dataset class in a .py file
# to the datasets folder and then listing it in the datasets/__init__.py file.
#
_C.DATASET.NAME = ""




_C.WANDB = CN()

# (optional)
# This project name will be used in weights and biases.
# In case not given, the run will not be logged in W&B.
#
_C.WANDB.PROJECT_NAME = ""




_C.TENSORBOARD = CN()

# (optional)
# This project name will be used in tensorboard.
# In case not given, the run will not be logged.
#
_C.TENSORBOARD.PROJECT_NAME = ""




_C.CHECKPOINT = CN()

# (required)
# The name of the folder to store the intermediate and the best
# checkpoints, this folder will be inside the run folder.
#
_C.CHECKPOINT.OUTPUT_FOLDER = "checkpoints"

# (required)
# Store the checkpoint every such epochs.
#
_C.CHECKPOINT.SAVE_EVERY = 10




_C.OPTIMIZER = CN()

# (required)
# The optimizer to use. Currently supported: "SGD", and "Adam".
#
_C.OPTIMIZER.NAME = "SGD"

# (required)
# The momentum in case required by the optimizer.
#
_C.OPTIMIZER.MOMENTUM = 0.9

# (required)
# The learning rate to be used by the optimizer.
#
_C.OPTIMIZER.LR = 0.001

# (optional)
# To update the learning rate at certain epochs (ex. [100, 120, 140])
# by multiplying the learning rate by _C.OPTIMIZER.GAMMA
#
_C.OPTIMIZER.MILESTONES = []

# (optional)
# The multiplying factor used to update the learning rate
# at each milestone.
#
_C.OPTIMIZER.GAMMA = 0.1

# (optional)
# The class weights to use in case weighted loss is desired and
# the selected loss funcation provides support for class weights.
# Only supported with "CrossEntropyLoss" currently.
#
_C.OPTIMIZER.WEIGHTS = []

# (optional)
# The loss function to use. Currently supported: "MSELoss", and "CrossEntropyLoss".
#
_C.OPTIMIZER.LOSS = ""




_C.TRAIN = CN()

# (required)
# The batch size to use in training.
#
_C.TRAIN.BATCH_SIZE = 32

# (required)
# The maximum number of epochs to reach either from 0,
# or from the checkpoint epoch in case of resuming a run.
#
_C.TRAIN.MAX_EPOCH = 300

# (optional)
#
_C.TRAIN.SHUFFLE = True




_C.EVAL = CN()

# (required)
# The batch size to use in evaluation.
#
_C.EVAL.BATCH_SIZE = 32

# (required)
# Run evaluation every such number of epochs.
# At this epoch, a checkpoint will be stored in case
# the epoch results with the best evaluation metrics.
#
_C.EVAL.RUN_EVERY = 2

# (optional)
#
_C.EVAL.SHUFFLE = True




_C.TEST = CN()

# (required)
# The batch size to use in testing.
#
_C.TEST.BATCH_SIZE = 32




def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values for my_project.
    """
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
