import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.utils import save_image
from utils.module import Generator, Discriminator
from utils.train_dataloader import TrainDataset
import logging
import os
from torch.utils.tensorboard import SummaryWriter

# Set up a logger
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# Set up TensorBoard writer
writer = SummaryWriter('logs')

# Check if there are available GPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name(device)
logger.info(f"Using device:{device}")

# Hyperparameters
batch_size = 12
lr = 1e-4
num_epochs = 1000
scale_factor = 2
root_dir = 'data/DCISIBC/train/DCISIBC_train_HR'

# Load dataset
train_dataset = TrainDataset(root_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
logger.info(f"The dataset has been loaded, with a total of {len (train_loader)} groups")

# Define generator and discriminator
generator = Generator(scale_factor=scale_factor).to(device)
discriminator = Discriminator().to(device)

# Load pretrained weights
# generator.load_state_dict(torch.load('modules/generator.pth'))
# discriminator.load_state_dict(torch.load('modules/discriminator.pth'))
# logger.info(f"Generator and discriminator weights have been loaded")

# Define loss function and optimizer
content_loss = nn.MSELoss().to(device)
adversarial_loss = nn.BCELoss().to(device)
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Load Vgg19
vgg = models.vgg19(pretrained=True).to(device)
for param in vgg.parameters():
    param.requires_grad = False
content_loss_vgg = nn.MSELoss().to(device)
logger.info(f"Weight loading completed")

# Train
for epoch in range(num_epochs):
    for i, (lr_images, hr_images) in enumerate(train_loader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)

        generated_hr_images = generator(lr_images)
        content_loss_value = content_loss(hr_images, generated_hr_images)
        adversarial_loss_value = adversarial_loss(discriminator(generated_hr_images), torch.ones(batch_size,1).to(device))

        high_res_vgg = vgg(hr_images)
        generated_high_res_vgg = vgg(generated_hr_images)
        content_loss_vgg_value = content_loss_vgg(high_res_vgg, generated_high_res_vgg)

        generator_loss = content_loss_value + 1e-3 * adversarial_loss_value + content_loss_vgg_value
        generator_loss /= 2
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        real_output = discriminator(hr_images)
        fake_output = discriminator(generated_hr_images.detach()).detach()
        real_loss = adversarial_loss(real_output, torch.ones(batch_size,1).to(device))
        fake_loss = adversarial_loss(fake_output, torch.zeros(batch_size,1).to(device))

        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        writer.add_scalar('Generator/Total Loss', generator_loss.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Generator/Content Loss', content_loss_value.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Generator/Adversarial Loss', adversarial_loss_value.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Generator/VGG Loss', content_loss_vgg_value.item(), epoch * len(train_loader) + i)
        writer.add_scalar('Discriminator/Loss', discriminator_loss.item(), epoch * len(train_loader) + i)

        # 输出训练信息
        if (i + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, "
                        f"G Loss: {generator_loss.item():.4f}, D Loss: {discriminator_loss.item():.4f}")

    # 保存模型和生成图像
    with torch.no_grad():
        generator.eval()
        if epoch == 0 or (epoch + 1) % 50 == 0:
            hr_images = next(iter(train_loader))[1][:2].to(device)
            lr_images = nn.functional.interpolate(hr_images, scale_factor=1/scale_factor, mode='bicubic')
            fake_hr_images = generator(lr_images)
            save_image(hr_images, f'output/hr_images_{epoch+1}.png')
            save_image(fake_hr_images, f'output/fake_hr_images_{epoch+1}.png')
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f'weight/generator_{epoch+1}.pth')
            torch.save(discriminator.state_dict(), f'weight/discriminator_{epoch+1}.pth')