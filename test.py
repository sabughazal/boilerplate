"""
Sultan Abughazal
"""
import os
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from configs import get_cfg_defaults
from models import MODELS
from datasets import DATASETS
from utils import load_checkpoint

## ARGUMENTS
parser = argparse.ArgumentParser(description="Test wavefronts decoding.")
parser.add_argument('--dataset-root', required=False, type=str, help="The path to the dataset root directory.") # required
parser.add_argument('--run-path', default=None, type=str, help="The path to the run to use in the testing. This is prioritized over --config, and --checkpoint.")
parser.add_argument('--config', default=None, type=str, help="The path to the config file. Will not be used if --run-path is given.")
parser.add_argument('--checkpoint', default=None, type=str, help="The path to the checkpoint to use in testing. Will not be used if --run-path is given.")
parser.add_argument('--gpu', default="cuda", type=str, help="The label of the GPU to use.")
args = parser.parse_args()

## VARIABLES
DEVICE = args.gpu if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = args.config
CHECKPOINT_PATH = args.checkpoint

assert os.path.exists(args.dataset_root)
if args.run_path:
    assert os.path.exists(args.run_path)
    CONFIG_PATH = os.path.join(args.run_path, "config.yml")
    CHECKPOINT_PATH = os.path.join(args.run_path, "checkpoints", "chkpt_best.pt")
assert os.path.exists(CONFIG_PATH)
assert os.path.exists(CHECKPOINT_PATH)

## CONFIGURATION
cfg = get_cfg_defaults()
cfg.merge_from_file(CONFIG_PATH)
cfg.freeze()

## FUNCTIONS
def log(line: str):
    print(line)

log("INFO: Running with the following arguments:\n{}".format(args))
log("INFO: Running with the following configuration:\n{}".format(cfg))
log("INFO: Using device: {}".format(DEVICE))



# prepare datasets and dataloaders
# 
def build_dataloader(cfg):
    dataset = DATASETS[cfg.DATASET.NAME](args.dataset_root, 'test', cfg=cfg)
    dataloader =  DataLoader(dataset=dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True)
    return dataloader, dataset


# prepare the model
# 
def build_model(cfg, device):
    return MODELS[cfg.MODEL.NAME](cfg).to(device)


# testing function
# 
def test(model, dataloader):
    """"""
    all_outputs = None
    all_targets = None
    with torch.no_grad():
        itr = tqdm(dataloader, total=len(dataloader), desc="Test", ncols=150)
        for inputs, targets in itr:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            
            if all_outputs is None:
                all_outputs = outputs.cpu().numpy()
                all_targets = targets.cpu().numpy()
            else:
                all_outputs = np.row_stack([all_outputs, outputs.cpu().numpy()])
                all_targets = np.row_stack([all_targets, targets.cpu().numpy()])

    return all_outputs, all_targets



# main
# 
def main(args, cfg):

    test_loader, test_dataset = build_dataloader(cfg)
    log("INFO: Testing dataset has {:,} samples.".format(len(test_dataset)))

    model = build_model(cfg, DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("INFO: The model has {:,} trainable parameters.".format(num_params))

    _ = load_checkpoint(CHECKPOINT_PATH, model)

    all_outputs, all_targets = test(model, test_loader)

    print("Labels shape:", all_targets.shape)
    print("Predictions shape:", all_outputs.shape)
    
    # calculate and print metrics
    loss_per_sample = np.mean((all_targets - all_outputs)**2, axis=1)
    mean_sq_error = np.mean(loss_per_sample)
    print("\nMSE: {}".format(mean_sq_error))



if __name__ == "__main__":
    main(args, cfg)