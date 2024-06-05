"""
Sultan Abughazal
"""
import os
import torch
import shutil
import logging

from tqdm import tqdm
from datetime import datetime

from configs import get_cfg_defaults
from torch.utils.data import DataLoader
from sabgbp.utils.utils import save_checkpoint, load_checkpoint

from sabgbp.models import MODELS
from sabgbp.datasets import DATASETS


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="The path to the config file.")
    parser.add_argument(
        '--dataset-root',
        type=str,
        required=True,
        help="The path to the dataset root directory.")

    # optional arguments
    parser.add_argument(
        '--run-name',
        type=str,
        default=datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"),
        help="The run name to use as a label in file names and in W&B.")
    parser.add_argument(
        '--output-folder',
        type=str,
        default="runs",
        help="The name of the folder to store information about the runs.")
    parser.add_argument(
        '--patience',
        type=int,
        default=None,
        help="Stop the training if the evaluation loss does not decrease in this many epochs.")
    parser.add_argument(
        '--gpu',
        type=str,
        default="cuda",
        help="The label of the GPU to use.")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="The path to the checkpoint to resume from.")
    parser.add_argument(
        '--resume',
        action="store_true",
        help="A flag to resume from a given checkpoint.")

    return parser.parse_args()



# prepare datasets and dataloaders
#
def build_dataloaders(cfg, dataset_root):
    train_dataset = DATASETS[cfg.DATASET.NAME](dataset_root, 'train', cfg=cfg)
    eval_dataset = DATASETS[cfg.DATASET.NAME](dataset_root, 'eval', cfg=cfg)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=cfg.TRAIN.SHUFFLE)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=cfg.EVAL.SHUFFLE)

    return train_loader, eval_loader, train_dataset, eval_dataset



# prepare the model
#
def build_model(cfg, device='cpu'):
    return MODELS[cfg.MODEL.NAME](cfg).to(device)

def build_optimizer(cfg, model):
    if cfg.OPTIMIZER.NAME == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM)
    elif cfg.OPTIMIZER.NAME == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR)
    return optimizer

def build_scheduler(cfg, optimizer):
    scheduler = None
    if len(cfg.OPTIMIZER.MILESTONES):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.OPTIMIZER.MILESTONES, gamma=cfg.OPTIMIZER.GAMMA)
    return scheduler

def build_criterion(cfg, device='cpu'):
    if cfg.OPTIMIZER.LOSS == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif cfg.OPTIMIZER.LOSS == "CrossEntropyLoss":
        class_weights = None
        if len(cfg.OPTIMIZER.WEIGHTS):
            class_weights = torch.Tensor(cfg.OPTIMIZER.WEIGHTS).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
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
def main(args):
    assert os.path.exists(args.config), "Configuration file does not exist!"
    assert os.path.exists(args.dataset_root), "Dataset root does not exist!"
    if args.resume:
        assert os.path.exists(args.checkpoint), "Resume checkpoint does not exist!"

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    if args.run_name.strip() == "":
        args.run_name = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    ## VARIABLES
    DEVICE = args.gpu if torch.cuda.is_available() else 'cpu'
    USE_WANDB = cfg.WANDB.PROJECT_NAME != ""
    USE_TENSORBOARD = cfg.TENSORBOARD.PROJECT_NAME != ""
    OUTPUT_PATH = os.path.join(args.output_folder, args.run_name)
    os.makedirs(OUTPUT_PATH, exist_ok=False)
    shutil.copy(args.config, os.path.join(OUTPUT_PATH, "config.yml"))


    # prepare logger
    #
    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s ## %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    log_fpath = os.path.join(OUTPUT_PATH, "log.txt")
    file_handler = logging.FileHandler(log_fpath)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(logging.INFO)

    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    log.info("Running with the following arguments:\n{}".format(args))
    log.info("Running with the following configuration:\n{}".format(cfg))
    log.info(f"Using device: {DEVICE}")


    # set up weights and biases
    #
    if USE_WANDB:
        import wandb
        wandb.init(
            project=cfg.WANDB.PROJECT_NAME,
            name=args.run_name,
            config={**vars(args),**cfg,},
        )


    # set up tensorboard
    #
    if USE_TENSORBOARD:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir="tb", comment=f"{cfg.TENSORBOARD.PROJECT_NAME}:{args.run_name}")
        # writer = tf.train.SummaryWriter('%s/%s' % (FLAGS.log_dir, run_var), sess.graph_def)
        # config={**vars(args),**cfg,}


    # set up dataloaders
    #
    train_loader, eval_loader, train_dataset, eval_dataset = build_dataloaders(cfg, args.dataset_root)
    log.info("Training dataset has {:,} samples.".format(len(train_dataset)))
    log.info("Evaluation dataset has {:,} samples.".format(len(eval_dataset)))

    model = build_model(cfg, DEVICE)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    criterion = build_criterion(cfg, DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("The model has {:,} trainable parameters.".format(num_params))
    log.info("The model has the following structure:\n{}".format(model))



    # main training loop
    #
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
        if USE_TENSORBOARD:
            tb_writer.add_scalar("train_loss", train_loss, epoch)

        epochs_since_best += 1
        if (epoch+1) % cfg.EVAL.RUN_EVERY == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH:
            model.eval()
            eval_loss = evaluate(model, eval_loader, criterion, DEVICE, eval_losses)
            if USE_WANDB:
                wandb.log({"eval_loss": eval_loss, "epoch": epoch})
            if USE_TENSORBOARD:
                tb_writer.add_scalar("eval_loss", eval_loss, epoch)

            if scheduler:
                scheduler.step()
                log.info("Using a learning rate of {}.".format(scheduler.get_last_lr()[0]))

            if eval_loss < best_eval_loss:
                save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler, is_best=True)
                best_eval_loss = eval_loss
                epochs_since_best = 0
                log.info("Best checkpoint saved at epoch {}!".format(epoch+1))

            log.info("Eval loss: {:.4f}; Best eval loss: {:.4f}.".format(eval_loss, best_eval_loss))

        if (epoch+1) % cfg.CHECKPOINT.SAVE_EVERY == 0 or (epoch+1) == cfg.TRAIN.MAX_EPOCH:
            save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler)
            log.info("Checkpoint saved at epoch {}!".format(epoch+1))

        if args.patience and epochs_since_best >= args.patience:
            log.info("Early stopping triggered; eval loss did not go lower than {:.4f} in the last {} epochs.".format(best_eval_loss, args.patience))
            save_checkpoint(cfg, model, optimizer, epoch+1, best_eval_loss, output_path=OUTPUT_PATH, scheduler=scheduler)
            log.info("Checkpoint saved at epoch {}!".format(epoch+1))
            break



if __name__ == "__main__":
    args = parse_args()
    main(args)
