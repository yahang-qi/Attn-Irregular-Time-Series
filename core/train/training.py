import os
import csv

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchmetrics

from core.data.dataset import PysioNet2012Dataset
from core.data import ImbalancedDatasetSampler
from core.data import recursive_device

from core.models import PhysioFormer
from core.train.lr_scheduler import NoamLR


def train(configs: dict):
    
    # Current time in the format of YearMonthDay-HourMinuteSecond
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    device = torch.device(configs['model_config']['device']) if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using {device}.')
    
    data_configs = configs['data_config']
    train_dataset = PysioNet2012Dataset(data_configs['train_dataset_path'])
    #train_loader = DataLoader(
    #    dataset=train_dataset,
    #    sampler=ImbalancedDatasetSampler(train_dataset),
    #    batch_size=data_configs['train_batch_size'],
    #    shuffle=False,
    #    num_workers=4,
    #    pin_memory=True,
    #)
 
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=data_configs['train_batch_size'],
        shuffle=True,
    )
    
    test_dataset = PysioNet2012Dataset(data_configs['test_dataset_path'])
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=data_configs['test_batch_size'],
        shuffle=False,
    )
    
    model_configs = configs['model_config']
    model = PhysioFormer(**model_configs).to(device)
    
    optimizer_configs = configs['optimizer_config']
    
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamLR(
        optimizer, 
        model_size=model_configs['embed_dim'], 
        factor=optimizer_configs['factor'],
        warmup_steps=optimizer_configs['warmup_steps'],
    )

    criterion = nn.BCELoss(weight=torch.tensor(10.0).to(device))
    num_epochs = optimizer_configs['num_epochs'] 

    # Metric instances
    auroc = torchmetrics.AUROC(task='binary').to(device)
    auprc = torchmetrics.AveragePrecision(task='binary').to(device)
    
    # Initialize the best AUROC
    best_auroc = 0.0
    
    # Define folder names with the current timestamp
    if not os.path.exists('./train_history'):
        os.makedirs('./train_history', exist_ok=True)
    timestamp_dir = f'./train_histroty_{timestamp}'
    os.makedirs(timestamp_dir, exist_ok=True)
    
    log_dir = os.path.join(timestamp_dir, "logs")
    ckpts_dir = os.path.join(timestamp_dir, "ckpts")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpts_dir, exist_ok=True)
        
    with open(os.path.join(log_dir, 'logs.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Step', 'Loss', 'Learning Rate'])

        step = 0
        
        for epoch in range(num_epochs):
            model.train()
            print(f'Epoch [{epoch+1}/{num_epochs}], training:')
            
            for data in train_loader:
                recursive_device(data, device)
                preds = model(data)
                loss = criterion(preds, data['Outcome'])
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        
                current_lr = scheduler.get_last_lr()[0] 
                
                writer.writerow([epoch+1, step+1, loss.item(), current_lr])
                file.flush()
                step += 1

                # Print loss and learning rate
                print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Learning Rate: {current_lr:.2e}')
                    
            # Evaluation phase
            if epoch%1 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], eval:')
                model.eval()
                with torch.no_grad():
                    for data in test_loader:
                        recursive_device(data, device)
                        preds = model(data)
                        auroc.update(preds, data['Outcome'].int())
                        auprc.update(preds, data['Outcome'].int())
                        
                # Get metric results
                epoch_auroc = auroc.compute()
                epoch_auprc = auprc.compute()
                print(f'Epoch [{epoch + 1}/{num_epochs}], AUROC: {epoch_auroc:.4f}, AUPRC: {epoch_auprc:.4f}')
                
                # Check if this is the best AUROC and save model if it is
                if epoch_auroc > best_auroc:
                    best_auroc = epoch_auroc
                    torch.save(model.state_dict(), os.path.join(ckpts_dir, f'best_model_{epoch+1}.pth'))
                    print(f"Saved new best model with AUROC: {best_auroc:.4f}")

                # Reset metrics for the next epoch
                auroc.reset()
                auprc.reset()
