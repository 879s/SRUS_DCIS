import os
from torchsummary import summary
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils.module import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# model loading
model = Generator()
model_dict = torch.load('weight/best.pth')
model.load_state_dict(model_dict)
model.eval()
model = model.to(device)

# image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# image test
img = Image.open('data/DCISIBC/test/test.png')
img = transform(img)
img = img.unsqueeze(0)
img = img.to(device)
with torch.no_grad():
    output = model(img)

# save image
output = output.squeeze(0).cpu().detach().numpy()
output = (output + 1) / 2.0 * 255.0
output = output.clip(0, 255).astype('uint8')
output = Image.fromarray(output.transpose(1, 2, 0))
output.save('output/test.png')

