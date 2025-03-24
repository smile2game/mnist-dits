from config import *
from torch.utils.data import DataLoader
from dataset import MNIST
from diffusion import forward_add_noise
import torch 
from torch import nn 
import os
from dit import DiT
import time
from tqdm import tqdm

DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备

dataset=MNIST() # 数据集加载 
print(f"加载数据集,dataset is {dataset}") 
model=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE) # 模型加载 
print(f"加载模型,model is {model}")

try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimzer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器

loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

'''
训练模型
'''

EPOCH=500
BATCH_SIZE=1000

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器
# # 打印 dataloader 的基本信息
# print(f"数据集大小: {len(dataset)}")  # 检查数据集的总大小
# print(f"每个 epoch 的迭代次数: {len(dataloader)}")  # 检查每个 epoch 的迭代次数

# # 遍历 dataloader 并打印数据情况
# for epoch in range(EPOCH):
#     print(f"\nEpoch {epoch + 1}/{EPOCH}")
#     for batch_idx, (imgs, labels) in enumerate(dataloader):
#         print(f"\nBatch {batch_idx + 1}/{len(dataloader)}")
#         print(f"imgs.shape: {imgs.shape}")  # 打印图片的形状
#         print(f"labels.shape: {labels.shape}")  # 打印标签的形状
#         print(f"imgs[0] (第一张图片):\n{imgs[0]}")  # 打印第一张图片的数据
#         print(f"labels[0] (第一个标签): {labels[0]}")  # 打印第一个标签的值

model.train()

start_all = time.time()
iter_count=0
for epoch in tqdm(range(EPOCH),desc = "Epochs"):
    epoch_start = time.time()
    # for imgs,labels in tqdm(dataloader,decs = "Batches",leave = False):
    for batch_idx, (imgs, labels) in tqdm(enumerate(dataloader),desc = "Batches",leave = False):
        # print(f"imgs.shape is {imgs.shape},labels.shape is {labels.shape},batch_idx is {batch_idx}")
        iter_start = time.time()
        x=imgs*2-1 # 图像的像素范围从[0,1]转换到[-1,1],和噪音高斯分布范围对应
        t=torch.randint(0,T,(imgs.size(0),))  # 为每张图片生成随机t时刻
        y=labels
        
        x,noise=forward_add_noise(x,t) # x:加噪图 noise:噪音
        pred_noise=model(x.to(DEVICE),t.to(DEVICE),y.to(DEVICE))

        loss=loss_fn(pred_noise,noise.to(DEVICE))
        
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        
        if iter_count%1000==0:
            print('epoch:{} iter:{},loss:{},save!'.format(epoch,iter_count,loss))
            torch.save(model.state_dict(),'.model.pth')
            os.replace('.model.pth','model.pth')
        iter_count+=1
        iter_end = time.time()
        # print(f"iter_count is {iter_count},cost time {iter_end-iter_start}s")
    epoch_end = time.time()
    print(f"epoch={epoch},cost time {epoch_end-epoch_start}s")
end_all = time.time()
print(f"finish all cost {end_all - start_all}!")