import torch
from torchvision.models import densenet
import train_densenet
from train_densenet import cxr_net
import torchvision.transforms as transforms
from PIL import Image

normalize = transforms.Normalize(mean=[0.449], #[0.485, 0.456, 0.406],
                                 std=[0.226]) #[0.229, 0.224, 0.225]),

model = cxr_net('densenet121', pretrained=True)
model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/out256x256/model_epoch24.pt'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),normalize])

img = Image.open('/home/64f/cxr/cxr_classification/36c192b5-e5f11e39-3d2677d2-18ee255c-9958a698.jpg')  # Load image as PIL.Image
x = transform(img)  # Preprocess image
x = x.unsqueeze(0)  # Add batch dimension

output = model(x)  # Forward pass
pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
print('Image predicted as ', pred)

