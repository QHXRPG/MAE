import torch.nn as nn
import torch
import numpy as np



def pos_coding(patch_num, bed_dim):
    patch = int(patch_num**0.5)
    a = np.arange(patch).reshape(1,patch)
    b = np.arange(patch).reshape(1,patch)
    omega = 1/(10000**((np.arange(bed_dim//4).reshape(-1))/256.))
    out1 = np.einsum('m,d->md',np.stack(np.meshgrid(a,b)).reshape(2,1,patch,patch)[0].reshape(-1),omega)
    out1_sin_cos = np.concatenate([np.sin(out1),np.cos(out1)],axis=1)
    out2 = np.einsum('m,d->md',np.stack(np.meshgrid(a,b)).reshape(2,1,patch,patch)[1].reshape(-1),omega)
    out2_sin_cos = np.concatenate([np.sin(out2),np.cos(out2)],axis=1)
    output = np.concatenate([out1_sin_cos,out2_sin_cos],axis=1)
    return np.concatenate([np.zeros([1,bed_dim]),output],axis=0)

"""多头自注意力机制"""
class Attentione(nn.Module):
    def __init__(self,dim=1024,head_num=8,drop1=0.,drop2=0.):
        super(Attentione, self).__init__()
        self.linear = nn.Linear(dim,dim*3)
        self.W0 = nn.Linear(dim,dim)
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)
        self.d = (dim/head_num)**-0.5
    def forward(self,x):
        # x:(batch,N,C)
        batch,N,C = x.shape
        qkv = self.linear(x) #(batch,N,3C)
        qkv = self.drop1(qkv)
        QKV = qkv.view(batch,N,3,8,-1)
        QKV = QKV.permute(2,0,3,1,4)
        q,k,v = QKV[0],QKV[1],QKV[2]
        attention = nn.functional.softmax((q@k.transpose(-1,-2))/self.d,dim=-1)
        attention = attention @ v
        attention = attention.transpose(1,2)  #torch.Size([64, 197, 8, 96])
        attention = attention.reshape(batch,N,C)
        attention = self.W0(attention)
        attention = self.drop2(attention)
        return attention

"""MLP"""
class MLP(nn.Module):
    def __init__(self,dim):
        super(MLP, self).__init__()
        self.l = nn.Sequential(nn.Linear(dim,1500),
                               nn.GELU(),
                               nn.Dropout(0.2),
                               nn.Linear(1500,dim),
                               nn.Dropout(0.))
    def forward(self,x):
        return self.l(x)

"""Encoder Block"""
class Block(nn.Module):
    def __init__(self,drop_attention,drop_mlp,dim):
        super(Block, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = Attentione(dim=dim)
        self.drop_attention = nn.Dropout(drop_attention)
        self.mlp = MLP(dim=dim)
        self.drop_mlp = nn.Dropout(drop_mlp)
    def forward(self,x):
        y = self.layer_norm(x)
        y = self.attention(y)
        y = self.drop_attention(y)
        z = y + x
        k = self.layer_norm(z)
        k = self.mlp(k)
        k = self.drop_mlp(k)
        return z+k

class MAE(nn.Module):
    def __init__(self,patch_num,en_dim,de_dim,mask_rate):
        super(MAE, self).__init__()
        self.en_dim = en_dim
        self.de_dim = de_dim
        self.patch_num = patch_num
        self.mask_rate = mask_rate
        self.emdeding = nn.Sequential(nn.Conv2d(3,en_dim,16,16),
                                      nn.BatchNorm2d(en_dim))  #torch.Size([64, 1024, 14, 14])
        self.pos_encoder = nn.Parameter(torch.zeros(1,patch_num+1,en_dim),requires_grad=False)
        self.pos_decoder = nn.Parameter(torch.zeros(1,patch_num+1,de_dim),requires_grad=False)
        self.cls_token = nn.Parameter(torch.zeros(1,1,en_dim))
        self.noise = torch.rand(64,patch_num)  #torch.Size([64, 196])
        self.block_encoder = Block(drop_attention=0,drop_mlp=0,dim=en_dim)
        self.block_decoder = Block(drop_attention=0,drop_mlp=0,dim=de_dim)
        self.mask_patch = nn.Parameter(torch.rand(1,1,de_dim)) #torch.Size([1, 1, 512])
        self.de_embeding = nn.Linear(en_dim,de_dim)
        self.linear = nn.Linear(de_dim,768,bias=True)
        self.loss = nn.MSELoss()
    def make_mask(self):
        keep_nums = int(self.patch_num*(1-self.mask_rate))
        ids_shuffle = torch.argsort(self.noise,dim=1)   #torch.Size([64, 196])
        ids_restore = torch.argsort(ids_shuffle,dim=1)   #torch.Size([64, 196])
        ids_keep = ids_shuffle[:,:keep_nums]
        ids_keep = ids_keep.unsqueeze(-1).repeat(1,1,self.en_dim) #torch.Size([64, 49, 1024])
        mask = torch.ones(64,self.patch_num)
        mask[:,:keep_nums] = 0
        mask = torch.gather(mask,dim=1,index=ids_restore)
        return ids_keep, mask, ids_restore

    def encoder(self,imgs):
        x = (self.emdeding(imgs)).view(64,self.en_dim,-1)
        x = x.transpose(2,1) #torch.Size([64, 196, 1024])
        pos_encoder = pos_coding(self.patch_num, self.en_dim)  #([197, 1024])
        pos_encoder = (torch.from_numpy(pos_encoder).float().unsqueeze(0))#torch.Size([1, 197, 1024])
        self.pos_encoder.data = pos_encoder  #torch.Size([1, 197, 1024])
        x = x+self.pos_encoder[:,1:,:] #torch.Size([64, 196, 1024])
        cls_token = self.cls_token + self.pos_encoder[:,:1,:]
        cls_token = cls_token.repeat(64,1,1)  # torch.Size([64, 1, 1024])
        """
        ids_keep: torch.Size([64, 49, 1024])
        mask: torch.Size([64, 196])
        ids_restore: torch.Size([64, 196])
        """
        ids_keep, mask, ids_restore = self.make_mask()
        ids_restore_encoder = ids_restore.unsqueeze(-1).repeat(1,1,self.en_dim)
        x_mask = torch.gather(x,dim=1,index=ids_restore_encoder) #torch.Size([64, 49, 1024])
        x_mask = torch.cat([cls_token,x_mask],dim=1)  #torch.Size([64, 50, 1024])
        for i in range(12):
            x_mask = self.block_encoder(x_mask) #torch.Size([64, 50, 1024])
        return x_mask, mask, ids_restore

    def decoder(self,x_mask, mask, ids_restore):
        x = self.de_embeding(x_mask)  #torch.Size([64, 50, 512])
        mask_patch = self.mask_patch.repeat(64,147,1)  #torch.Size([64, 147, 512])
        all_patch = torch.cat([x[:,1:,:],mask_patch],dim=1)  #torch.Size([64, 196, 512])
        ids_restore = ids_restore.unsqueeze(-1).repeat(1,1,self.de_dim) #torch.Size([64, 196, 512])
        all_patch = torch.gather(all_patch,dim=1,index=ids_restore)
        all_patch = torch.cat([x[:,:1,:],all_patch],dim=1)  #torch.Size([64, 197, 512])
        pos_decoder = pos_coding(self.patch_num, self.de_dim)
        self.pos_decoder.data.copy_(torch.from_numpy(pos_decoder).float().unsqueeze(0))  #torch.Size([1, 197, 512])
        all_patch = all_patch+self.pos_decoder
        for i in range(12):
            all_patch = self.block_decoder(all_patch) #torch.Size([64, 197, 512])
        all_patch = self.linear(all_patch)  #torch.Size([64, 197, 768])
        all_patch = all_patch[:,1:,:]  #torch.Size([64, 196, 768])
        return all_patch, mask
    def make_patchs(self,imgs):
        B,C,H,W = imgs.shape
        patch_num = int(self.patch_num**0.5)
        patch_size = int(H/(int(self.patch_num**0.5)))
        imgs = imgs.reshape(B,C,patch_num,patch_size,patch_num,patch_size)  #torch.Size([64, 3, 14, 16, 14, 16])
        imgs = torch.einsum('abcdef->acedfb',imgs)  #torch.Size([64, 14, 14, 16, 16, 3])
        imgs = imgs.reshape(B,patch_num**2,-1)   #torch.Size([64, 196, 768])
        return imgs  #torch.Size([64, 196, 768])
    def Loss(self,imgs, all_patch, mask):
        """
        :param imgs: 原图 torch.Size([64, 196, 768])
        :param all_patch: decoder输出的图像块 #torch.Size([64, 196, 768])
        :param mask: 原图掩码顺序 torch.Size([64, 196])
        :return: Loss 损失函数
        """
        y = self.make_patchs(imgs)
        mask = mask.unsqueeze(-1)
        train_patch = all_patch * mask
        y = y * mask
        loss = self.loss(train_patch,y)
        return loss
    def forward(self,imgs):
        x_mask, mask, ids_restore = self.encoder(imgs)
        all_patch, mask = self.decoder(x_mask, mask, ids_restore)
        loss = self.Loss(imgs,all_patch,mask)
        return loss,x_mask,all_patch

imgs = torch.rand(64,3,224,224)
mae = MAE(196,1024,512,0.75)
loss,x_mask,all_patch = mae(imgs)