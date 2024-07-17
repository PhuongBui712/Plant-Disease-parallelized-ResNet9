import wandb
from dotenv import load_dotenv


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
    