import os
import torch

def save_checkpoint(cfg, model, optimizer, epoch, best_eval_loss, output_path, scheduler=None, is_best=False):
    checkpoints_folder = os.path.join(output_path, cfg.CHECKPOINT.OUTPUT_FOLDER)
    # filename = "chkpt_ep{:.0f}_{}.pt".format(epoch, datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"))
    filename = "chkpt_ep{:.0f}.pt".format(epoch)
    if is_best:
        filename = "chkpt_best.pt"

    os.makedirs(checkpoints_folder, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_eval_loss': best_eval_loss,
    }, os.path.join(checkpoints_folder, filename))

def load_checkpoint(p, model, optimizer=None, scheduler=None, device=None):
    checkpoint = torch.load(p, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint.keys():
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def count_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
