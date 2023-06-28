"""
Sultan Abughazal
"""
import os
import torch
import shutil
import argparse
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from configs import get_cfg_defaults
from models import MODELS
from datasets import DATASETS
from utils import save_checkpoint, load_checkpoint

## ARGUMENTS
parser = argparse.ArgumentParser(description="")
parser.add_argument('--config', required=True, type=str, help="The path to the config file.") # required
parser.add_argument('--dataset-root', required=False, type=str, help="The path to the dataset root directory.") # required
parser.add_argument('--run-name', default=datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"), 
                    type=str, help="The run name to use as a label in file names and in W&B.")
parser.add_argument('--output-folder', default="runs", type=str, help="The name of the folder to store information about the runs.")
parser.add_argument('--patience', default=None, type=int, help="Stop the training if the evaluation loss does not decrease in this many epochs.")
parser.add_argument('--gpu', default="cuda", type=str, help="The label of the GPU to use.")
parser.add_argument('--checkpoint', default=None, type=str, help="The path to the checkpoint to resume from.")
parser.add_argument('--resume', action="store_true", help="A flag to resume from a given checkpoint.")
args = parser.parse_args()

assert os.path.exists(args.config), "Configuration file does not exist!"
assert os.path.exists(args.dataset_root), "Dataset root does not exist!"
if args.resume:
    assert os.path.exists(args.checkpoint), "Checkpoint does not exist!"
if args.run_name.strip() == "":
    args.run_name = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

## CONFIGURATION
cfg = get_cfg_defaults()
cfg.merge_from_file(args.config)
cfg.freeze()

## VARIABLES
DEVICE = args.gpu if torch.cuda.is_available() else 'cpu'
USE_WANDB = cfg.WANDB.PROJECT_NAME != ""
OUTPUT_PATH = os.path.join(args.output_folder, args.run_name)
os.makedirs(OUTPUT_PATH, exist_ok=False)
shutil.copy(args.config, os.path.join(OUTPUT_PATH, "config.yml"))

## FUNCTIONS
def log(line: str):
    print(line)
    with open(os.path.join(OUTPUT_PATH, "log.txt"), 'a') as f:
        f.write(line+"\n")

log("INFO: Running with the following arguments:\n{}".format(args))
log("INFO: Running with the following configuration:\n{}".format(cfg))
log("INFO: Using device: {}".format(DEVICE))



# set up weights and biases
# 
if USE_WANDB:
    import wandb
    wandb.init(
        project=cfg.WANDB.PROJECT_NAME,
        name=args.run_name,
        config={
            'arguments': args,
            'configuration': cfg,
        },
    )



# prepare datasets and dataloaders
# 
def build_dataloaders(cfg):
    train_dataset = DATASETS[cfg.DATASET.NAME](args.dataset_root, 'train', cfg=cfg)
    eval_dataset = DATASETS[cfg.DATASET.NAME](args.dataset_root, 'eval', cfg=cfg)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=cfg.EVAL.SHUFFLE)

    return train_loader, eval_loader, train_dataset, eval_dataset



# prepare the model
# 
def build_model(cfg, device='cpu'):
    return MODELS[cfg.MODEL.NAME].to(device)

def build_optimizer(cfg, model):
    if cfg.OPTIMIZER.NAME == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM)
    elif cfg.OPTIMIZER.NAME == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR)
    return optimizer

def build_scheduler(cfg, optimizer):
    scheduler = None
    if len(cfg.OPTIMIZER.MILESTONES):
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.OPTIMIZER.MILESTONES, gamma=cfg.OPTIMIZER.GAMMA)
    return scheduler

def build_criterion(cfg, device='cpu'):
    if cfg.OPTIMIZER.LOSS == "MSELoss":
        criterion = nn.MSELoss()
    elif cfg.OPTIMIZER.LOSS == "CrossEntropyLoss":
        class_weights = None
        if len(cfg.OPTIMIZER.WEIGHTS):
            class_weights = torch.Tensor(cfg.OPTIMIZER.WEIGHTS).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    return criterion



# training function
# 
def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu', progress=None, train_losses=[]):
    """"""
    losses_sum = 0
    t = f"Epoch {progress[0]+1}/{progress[1]}" if progress else ""
    itr = tqdm(dataloader, total=len(dataloader), desc=t, ncols=150)
    for inputs, targets in itr:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses_sum += loss.item()
    
    train_loss = losses_sum / len(dataloader)
    train_losses.append(train_loss)

    return train_loss, outputs



# evaluation function
# 
def evaluate(model, dataloader, criterion, device='cpu', eval_losses=[]):
    """"""
    losses_sum = 0
    with torch.no_grad():
        itr = tqdm(dataloader, total=len(dataloader), desc="Evaluate", ncols=100)
        for inputs, targets in itr:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses_sum += loss.item()

    eval_loss = losses_sum / len(dataloader)
    eval_losses.append(eval_loss)

    return eval_loss



# main training loop
# 
def main(args, cfg):

    model = build_model(cfg, DEVICE)

    train_loader, eval_loader, train_dataset, eval_dataset = build_dataloaders(cfg)
    log("INFO: Training dataset has {:,} samples.".format(len(train_dataset)))
    log("INFO: Evaluation dataset has {:,} samples.".format(len(eval_dataset)))

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_criterion(cfg, DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("INFO: The model has {:,} trainable parameters.".format(num_params))
    log("INFO: The model has the following structure:\n{}".format(model))

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoint, model, optimizer, scheduler, DEVICE)

    train_losses = [] # not actually used
    eval_losses = [] # not actually used
    best_eval_loss = 1e10
    epochs_since_best = 0
    for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
        
        model.train()
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, 
                progress=(epoch, cfg.TRAIN.MAX_EPOCH), train_losses=train_losses)

        if USE_WANDB:
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        epochs_since_best += 1
        if (epoch+1) % cfg.EVAL.RUN_EVERY == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH:
            model.eval()
            eval_loss = evaluate(model, eval_loader, criterion, DEVICE, eval_losses)
            if USE_WANDB:
                wandb.log({"eval_loss": eval_loss, "epoch": epoch})
            log("INFO: Eval loss: {:.4f}; Best eval loss: {:.4f}.".format(eval_loss, best_eval_loss))

            if scheduler:
                scheduler.step()
                log("INFO: Using a learning rate of {}.".format(scheduler.get_last_lr()[0]))

            if eval_loss < best_eval_loss:
                save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler, is_best=True)
                best_eval_loss = eval_loss
                epochs_since_best = 0
                log("INFO: Best checkpoint saved at epoch {}!".format(epoch+1))

        if (epoch+1) % cfg.CHECKPOINT.SAVE_EVERY == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH:
            save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler)
            log("INFO: Checkpoint saved at epoch {}!".format(epoch+1))

        if args.patience and epochs_since_best >= args.patience:
            log("INFO: Early stopping triggered; eval loss did not go lower than {:.4f} in the last {} epochs.".format(best_eval_loss, args.patience))
            save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler)
            log("INFO: Checkpoint saved at epoch {}!".format(epoch+1))
            break


if __name__ == "__main__":
    main(args, cfg)