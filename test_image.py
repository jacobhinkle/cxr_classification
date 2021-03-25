import torch
from torch import nn
from torchvision.models import densenet
import torch.distributed as dist
import train_densenet
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from train_densenet import cxr_densenet
import torchvision.transforms as transforms
from PIL import Image
import os

model = cxr_densenet('densenet121', pretrained=True)
rank=0
world_size=1
local_rank=0

#gpunum = local_rank 
#device = torch.device('cuda', gpunum)
#model = model.to(device)


dist.get_world_size()
dist.init_process_group('nccl')
#model = DDP(model,device_ids=[gpunum],output_device=gpunum)


normalize = transforms.Normalize(mean=[0.449], #[0.485, 0.456, 0.406],
                                 std=[0.226]) #[0.229, 0.224, 0.225]),

model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/out2048x2048/model_epoch4.pt'))
model.eval()

transform = test_transform=transforms.Compose([transforms.ToTensor(),normalize])

img = Image.open('/home/64f/cxr/cxr_classification/36c192b5-e5f11e39-3d2677d2-18ee255c-9958a698.jpg')  # Load image as PIL.Image
x = transform(img)  # Preprocess image
x = x.unsqueeze(0)  # Add batch dimension

output = model(x)  # Forward pass
pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
print('Image predicted as ', pred)
