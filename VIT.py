import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torchvision.transforms as transforms
train_data = FashionMNIST(root="./",train=True,transform=transforms.ToTensor(),download=True)
train_loader = Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)  #torch.Size([64, 1, 28, 28])

"""多头自注意力机制"""
class Attention(nn.Module):
    def __init__(self,dim=768,head_num=8,drop1=0.,drop2=0.):
        super(Attention, self).__init__()
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

"""Encoder Block"""
class Encoder_block(nn.Module):
    def __init__(self,drop_attention,drop_mlp,dim):
        super(Encoder_block, self).__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = Attention()
        self.drop_attention = nn.Dropout(drop_attention)
        self.mlp = MLP(dim)
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

"""VIT"""
class VIT(nn.Module):
    def __init__(self,batchsize=64,dim=768,drop_pos=0.,drop_attention=0.,drop_mlp=0.,classes=10):
        super(VIT, self).__init__()
        self.cls_token = nn.Parameter(torch.zeros(batchsize,1,dim))
        self.embeding = nn.Sequential(nn.Conv2d(1,dim,2,2)) #torch.Size([64, 768, 14, 14])
        self.pos = nn.Parameter(torch.zeros(batchsize,197,dim),requires_grad=False)
        self.pos_drop = nn.Dropout(drop_pos)
        self.encoder_block = Encoder_block(drop_attention,drop_mlp,dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.mlphead = nn.Sequential(nn.Linear(dim,2000),
                                     nn.Tanh(),
                                     nn.Linear(2000,classes))
    def forward(self,x):
        x=(self.embeding(x)).flatten(2) #torch.Size([64, 768, 196])
        x = x.transpose(1,2) #torch.Size([64, 196, 768])
        x = torch.cat([self.cls_token,x],dim=1)  # torch.Size([64, 197, 768])
        x = self.pos + x # torch.Size([64, 197, 768])
        x = self.pos_drop(x) # torch.Size([64, 197, 768])
        for i in range(12):
            x = self.encoder_block(x)
        x = self.layer_norm(x)
        x = ((x.transpose(0,1))[0])
        return self.mlphead(x)
vit = VIT()
opt = torch.optim.Adam(vit.parameters(),lr=0.001)
loss = nn.CrossEntropyLoss()
def train(epoch,model,loader,optim,loss):
    model.train()
    for i in range(epoch):
        for j,(x,y) in enumerate(loader):
            y_p = model(x)
            l = loss(y_p,y)
            optim.zero_grad()
            l.backward()
            opt.step()
            print(l.item())
