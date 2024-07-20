import wandb
from dotenv import load_dotenv
from torch.optim import Optimizer


load_dotenv()


def setup_wandb(project_name: str, run_name: str, batch_size: int, epoch: int):
    """
    Sets up Weights & Biases (WandB) for experiment tracking.

    Args:
        project_name (str): The name of the WandB project.
        run_name (str): The name of the WandB run.
        batch_size (int): The batch size used for training.
        epoch (int): The current epoch number.
    """
    wandb.login()

    wandb.init(
        project=project_name,
        name=run_name,
        config={
            'epoch': epoch,
            'batch_size': batch_size
        },
    )
    

def get_lr(optimizer: Optimizer):
    """
    Returns the learning rate of the optimizer.

    Args:
        optimizer (Optimizer): The optimizer to get the learning rate from.

    Returns:
        float: The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']