import torch
from torch.utils.data import DataLoader
from dataset import MNIST
from dit import DiT
from inference import backward_denoise
from torchvision.utils import save_image
from torchvision.transforms import Resize
import os
from torch_fidelity import calculate_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model = DiT(img_size=28, patch_size=4, channel=1, emb_size=64, label_num=10, dit_num=3, head=4).to(DEVICE)
model.load_state_dict(torch.load('model.pth'))

# 定义生成图像函数
def generate_images(model, num_images, batch_size=100, device='cuda'):
    model.eval()
    generated_images = []
    with torch.no_grad():
        for _ in range(0, num_images, batch_size):
            current_batch_size = min(batch_size, num_images - len(generated_images))
            x = torch.randn(size=(current_batch_size, 1, 28, 28)).to(device)
            y = torch.randint(0, 10, (current_batch_size,)).to(device)
            steps = backward_denoise(model, x, y)
            final_img = steps[-1].to('cpu')
            final_img = (final_img + 1) / 2
            generated_images.append(final_img)
    return torch.cat(generated_images, dim=0)

# 加载验证集
val_dataset = MNIST(is_train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

# 获取验证集图像和标签
val_images = []
val_labels = []
for imgs, labels in val_dataloader:
    val_images.append(imgs)
    val_labels.append(labels)
val_images = torch.cat(val_images, dim=0)
val_labels = torch.cat(val_labels, dim=0)

# 生成图像
generated_images = generate_images(model, num_images=len(val_dataset), batch_size=1000, device=DEVICE)

# 保存图像函数（调整为 Inception V3 格式）
resize = Resize(299)
def save_images(images, labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, (img, label) in enumerate(zip(images, labels)):
        img = resize(img)
        img = img.repeat(3, 1, 1)
        save_path = os.path.join(save_dir, f"{label}_{i}.png")
        save_image(img, save_path)

# 保存图像
save_images(val_images, val_labels, 'val_images')
save_images(generated_images, val_labels, 'generated_images')

# 计算 FID
metrics = calculate_metrics(
    input1="path/to/generated/images",
    input2="path/to/real/images",
    cuda=True,  # Set to False if not using GPU
    fid=True    # Explicitly request FID calculation
)
fid_score = metrics["frechet_inception_distance"]
print(f"FID 分数: {fid_score}")