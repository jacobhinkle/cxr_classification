import torch
from torchvision.models import densenet
import train_densenet
from train_densenet import cxr_net
import torchvision.transforms as transforms
from PIL import Image

normalize = transforms.Normalize(mean=[0.449], #[0.485, 0.456, 0.406],
                                 std=[0.226]) #[0.229, 0.224, 0.225]),

model = cxr_net('densenet121', pretrained=True)
model.load_state_dict(torch.load('/home/64f/cxr/cxr_classification/saved_models/256/model_epoch11.pt'))
model.eval()

transform = transforms.Compose([transforms.ToTensor(),normalize])

img = Image.open('/home/64f/cxr/cxr_classification/53bdbaea-2fef8033-aca685ff-2924187a-b092c1cd.jpg')  # Load image as PIL.Image
x = transform(img)  # Preprocess image
x = x.unsqueeze(0)  # Add batch dimension

output = model(x)  # Forward pass
print(output)
sm = torch.nn.Softmax(dim=1) #softmax function for getting probabilities
pred = sm(output)  # Get predictions
print(pred)
