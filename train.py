from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from src.dataset import PlantDiseaseDataset
from src.resnet9.numba_resnet9 import NumbaResNet9


def train():
    device = 'cuda:0'

    # Load data
    data_path = './data/New-Plant-Diseases-Dataset/'

    train_dataset = PlantDiseaseDataset(data_path + 'train')
    val_dataset = PlantDiseaseDataset(data_path + 'valid')

    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = NumbaResNet9(3, 38)
    model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-5)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        
        train_loop = tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}', leave=True)
        for i, data in enumerate(train_loop):
            X, y = (_.cuda() for _ in data)
            
            y_pred = model(X)
            
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item()
            
            train_loop.set_postfix({'loss': running_loss / (i + 1)})