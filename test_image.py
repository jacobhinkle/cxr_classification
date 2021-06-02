import torch
from torchvision.models import densenet
import train_densenet
from train_densenet import cxr_net
import torchvision.transforms as transforms
from PIL import Image
import mimic_cxr_jpg

normalize = transforms.Normalize(mean=[0.449], #[0.485, 0.456, 0.406],
                                 std=[0.226]) #[0.229, 0.224, 0.225]),

model = cxr_net('densenet121', pretrained=True)
model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/saved_models/1024/model_epoch17.pt'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),normalize])

img = Image.open('/home/64f/cxr/cxr_classification/test_images/1024/6044b262-5b22b18f-be11ace3-9097688f-e41c07e7.jpg')  # Load image as PIL.Image
x = transform(img)  # Preprocess image
x = x.unsqueeze(0)  # Add batch dimension
output = model(x)  # Forward pass
pred = torch.sigmoid(output)  # Get predictions
print(pred)
labels = []
for i in range(14):
    if pred[0,i] >= 0.5:
        labels.append(mimic_cxr_jpg.chexpert_labels[i])
print(labels)
if labels == []:
    labels.append('None')
