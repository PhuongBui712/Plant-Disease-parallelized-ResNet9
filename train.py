import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.dataset import PlantDiseaseDataset
from src.resnet9.numba_resnet9 import NumbaResNet9
from src.utils import *
from src.metrics import accuracy


def train(batch_size: int,
          epochs: int,
          max_lr: float,
          weight_decay: float,
          grad_clip: float,
          optimizer: Optimizer):
    device = 'cuda:0'

    # Load data
    data_path = './data/New-Plant-Diseases-Dataset/'

    train_dataset = PlantDiseaseDataset(data_path + 'train')
    val_dataset = PlantDiseaseDataset(data_path + 'valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = NumbaResNet9(3, 38)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # Loss, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.AdamW(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                            max_lr,
                                            epochs=epochs,
                                            steps_per_epoch=len(train_dataloader))

    # experiment monitor
    setup_wandb(project_name='Plant Diease Identification',
                run_name='Numba ResNet9',
                batch_size=batch_size,
                epoch=epochs)
    STEP_PER_LOG = 10

    # Training loop
    torch.cuda.empty_cache()
    batch_count, num_log = 0, 1

    for epoch in range(epochs):
        train_running_loss, train_acc = 0.0, 0.0
        logging_dict = {}
        
        # Train
        model.train() 
        
        print(f'Epoch {epoch + 1}/{epochs}')
        train_loop = tqdm(train_dataloader, desc=f'{"Train":^7}', leave=True)
        for i, data in enumerate(train_loop):
            # load data to cuda
            X, y = (_.cuda() for _ in data)
            
            # compute y_pred
            y_pred = model(X)
            
            # loss
            loss = criterion(y_pred, y)
            loss.backward()
            
            ## gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # update lr
            scheduler.step()
            
            # update loss
            train_running_loss += loss.item()
            train_acc += accuracy(y_pred, y)
            
            logging_dict = {'loss': train_running_loss / (i + 1),
                            'accuracy': train_acc / (i + 1)}
            
            # update progress bar
            train_loop.set_postfix(logging_dict)
            
            # wandb logging
            batch_count += 1
            if batch_count // STEP_PER_LOG == num_log or i == len(train_dataloader) - 1:
                logging_dict['epoch'] = batch_count / len(train_dataloader)
                logging_dict['learning rate'] = get_lr(optimizer)
                
                wandb.log({f'train/{k}': v for k, v in logging_dict.items()}, step=batch_count)
                
                num_log += 1
                
        # Evaluate
        model.eval()
        val_running_loss, val_acc = 0.0, 0.0
        val_loop = tqdm(val_dataloader, desc=f"{'Eval':^7}", leave=True)
        for i, data in enumerate(val_loop):
            X, y = (_.to(device) for _ in data)

            y_pred = model(X)

            loss = criterion(y_pred, y)

            val_running_loss += loss.item()
            val_acc += accuracy(y_pred, y)

            logging_dict = {
                'loss': val_running_loss / (i + 1),
                'accuracy': val_acc / (i + 1)
            }
            val_loop.set_postfix(logging_dict)

        wandb.log({
            'train/epoch': epoch + 1,
            'eval/loss': val_running_loss / len(val_dataloader),
            'eval/accuracy': val_acc / len(val_dataloader)
        })


if __name__ == '__main__':
    train()