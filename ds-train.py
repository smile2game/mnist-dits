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
import deepspeed
import wandb  # 导入 wandb

def init_deepspeed(model, optimizer, args):
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=args['deepspeed_config']  # DeepSpeed 配置文件路径
    )
    return model_engine, optimizer

DEVICE='cuda' if torch.cuda.is_available() else 'cpu' # 设备
dataset=MNIST() # 数据集加载 
print(f"加载数据集,dataset is {dataset}") 
model=DiT(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(DEVICE) # 模型加载 
print(f"加载模型,model is {model}")

try:    # 加载模型
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)   # 优化器
loss_fn=nn.L1Loss() # 损失函数(绝对值误差均值)

'''
训练模型
'''
# 初始化 DeepSpeed
args = {
    "deepspeed_config": "ds_config.json"  # DeepSpeed 配置文件路径
}
model_engine, optimizer = init_deepspeed(model, optimizer, args)

EPOCH=500
BATCH_SIZE=1000

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)    # 数据加载器


model.train()
#初始化 wandb
wandb.init(project="mnist-diffusion", name="dit-training-run")

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
        
        optimizer.zero_grad()
        # loss.backward()
        model_engine.backward(loss)
        # optimizer.step()
        model_engine.step()
        # 记录损失到 wandb
        wandb.log({"loss": loss.item(), "epoch": epoch, "batch": batch_idx})
        
        if iter_count%1000==0:
            print('epoch:{} iter:{},loss:{},save!'.format(epoch,iter_count,loss))
            torch.save(model.state_dict(),f'model_{iter_count}.pth')
            # os.replace('.model.pth','model.pth')
        iter_count+=1
        iter_end = time.time()
        # print(f"iter_count is {iter_count},cost time {iter_end-iter_start}s")
    epoch_end = time.time()
    print(f"epoch={epoch},cost time {epoch_end-epoch_start}s")
end_all = time.time()
print(f"finish all cost {end_all - start_all}!")

# 完成 wandb 运行
wandb.finish()