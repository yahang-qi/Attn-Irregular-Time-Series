import torch

class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, model_size, factor, warmup_steps, last_epoch=-1):
        self.model_size = model_size
        self.factor = factor
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step = max(1, self.last_epoch)
        scale = self.factor * (self.model_size ** (-0.5) * 
                               min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]
