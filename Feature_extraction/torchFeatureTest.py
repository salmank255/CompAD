

import torch
from torchsummary import summary
import torch.nn as nn

# Choose the `slow_r50` model 
model = torch.hub.load('facebookresearch/pytorchvideo:main', 'slow_r50', pretrained=True)

print(model)

# Set to GPU or CPU
device = "cuda"
model = model.eval()
model = model.to(device)

model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
print(model.classifier)

summary(model, (3,10, 256, 256))



