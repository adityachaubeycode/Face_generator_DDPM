import torch
import os 
import torchvision
import torch.optim
import torch.nn as nn
from torchvision.transforms import  transforms
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from timeit import default_timer as timer 
from model import Unet
from dataloader import get_loaders

data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def q_sample(x0, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod[t]).view(-1, 1, 1, 1)
    
    return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

data_path = 'cd'
dataset , dataloader = get_loaders(data_path=data_path, batch_size=32,  data_transform=data_transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DDPLighting(L.LightningModule):
    def __init__(self, T=1000):
        super().__init__()
        self.model = Unet()
        self.loss_fn = nn.MSELoss()
        self.T = T
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start , beta_end , steps=T)
        alphas = 1 - betas 
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

    def training_step(self, batch , batch_idx):
        img = batch
        t = torch.randint(0, self.T , (img.shape[0],), device=img.device)
        noise = torch.randn_like(img)
        xt = q_sample(x0=img, t=t, alphas_cumprod=self.alphas_cumprod, noise=noise)
        output = self.model(xt, t)
        loss = self.loss_fn(output, noise)
        self.log("train_Loss", loss, prog_bar=True, on_epoch=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        return optimizer

checkpoint_callback = ModelCheckpoint(
        dirpath='/teamspace/studios/this_studio/Face_generator_DDPM/checkpoints',
        filename='celebA_gen-{epoch:02d}',
        monitor='train_loss',
        mode='min',
        save_top_k=1  )

#Trainer class 
trainer = L.Trainer(max_epochs=1,
                    accelerator='gpu',
                    devices=1,
                    precision='16-mixed',
                    callbacks=[checkpoint_callback] )

start = timer()
model = DDPLighting()
trainer.fit(model=model, train_dataloaders=dataloader)
end = timer()

elapsed_min = (end - start) / 60
print(f'Total Training: {elapsed_min:.2f} min')