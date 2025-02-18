import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, kernel):
        super(conv_block, self).__init__()
        
        if type(kernel) is int:
            pad = kernel//2
        else:
            pad = [k//2 for k in kernel]
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, kernel):
        super(up_conv, self).__init__()
        
        if type(kernel) is int:
            pad = kernel//2
        else:
            pad = [k//2 for k in kernel]
        
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
        
class AttUNet(LightningModule):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, 
                 img_ch = 3,
                 output_ch = 1):
        
        super(AttUNet, self).__init__()
        
        self.n_channels = img_ch
        self.n_classes = output_ch
        self.save_hyperparameters()
        
        self.training_task = 'mae'
                
        n1 = 8
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        kernel = [3, 9]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0], kernel)
        self.Conv2 = conv_block(filters[0], filters[1], kernel)
        self.Conv3 = conv_block(filters[1], filters[2], kernel)
        self.Conv4 = conv_block(filters[2], filters[3], kernel)
        self.Conv5 = conv_block(filters[3], filters[4], kernel)

        self.Up5 = up_conv(filters[4], filters[3], kernel)
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3], kernel)

        self.Up4 = up_conv(filters[3], filters[2], kernel)
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2], kernel)

        self.Up3 = up_conv(filters[2], filters[1], kernel)
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1], kernel)

        self.Up2 = up_conv(filters[1], filters[0], kernel)
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0], kernel)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)
        
        self.dropout = nn.Dropout2d(p=0.6)
        
    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.dropout(e3)
        
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.dropout(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out
    
    def set_training_task(self, new_task):
        assert new_task in ['mae', 'seg']
        self.training_task = new_task
    
    def configure_optimizers(self): # change
        lr0 = 1e-4
        optimizer = torch.optim.Adam(self.parameters(), lr=lr0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-8) # cosineanneal was 30 for MAE
        return {"optimizer": optimizer,  "lr_scheduler": {"scheduler": scheduler}}
        # return {"optimizer": optimizer}
    
    def mae_step(self, batch):
        
        imgs, masks = batch
 
        masked_imgs = imgs.clone().detach()
        masked_imgs[:, 1:2, :, :][masks] = -1 # mask is applied to center channel
        # masked_imgs[torch.tile(masks, (1, 3, 1, 1))] = -1 # mask is applied to all channels
        
        pred = self(masked_imgs)

        loss = torch.sum((pred[masks] - imgs[:, 1:2, :, :][masks])**2) / torch.count_nonzero(masks) # calculate MSE
        return loss
        
    def seg_step(self, batch):
        
        imgs, masks = batch
        pred = self(imgs)
        loss = F.binary_cross_entropy_with_logits(pred, masks)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        
        if self.training_task == 'mae':
            loss = self.mae_step(batch)
        elif self.training_task == 'seg':
            loss = self.seg_step(batch)            
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        if self.training_task == 'mae':
            loss = self.mae_step(batch)
        elif self.training_task == 'seg':
            loss = self.seg_step(batch)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss
        
    
    
        
