#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


# In[ ]:


# Dataset Preparation
class SyntheticShapesDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=(128,128), shape_size=(20, 20), num_shapes=3):
        self.num_samples = num_samples
        self.image_size = image_size
        self.shape_size = shape_size
        self.num_shapes = num_shapes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a blank RGB image with random background color
        background_color = tuple(np.random.randint(0, 256, size=3))
        image = Image.new('RGB', self.image_size, color=background_color)
        draw = ImageDraw.Draw(image)

        # Draw random shapes on the image
        for _ in range(self.num_shapes):
            shape_type = np.random.choice(['circle', 'rectangle'])
            shape_color = tuple(np.random.randint(0, 256, size=3))
            shape_position = (np.random.randint(0, self.image_size[0]-self.shape_size[0]),
                              np.random.randint(0, self.image_size[1]-self.shape_size[1]))

            if shape_type == 'circle':
                draw.ellipse([shape_position, (shape_position[0]+self.shape_size[0], shape_position[1]+self.shape_size[1])], fill=shape_color)
            elif shape_type == 'rectangle':
                draw.rectangle([shape_position, (shape_position[0]+self.shape_size[0], shape_position[1]+self.shape_size[1])], fill=shape_color)

        # Convert image to tensor
        transform = transforms.ToTensor()
        image = transform(image)

        # Generate binary mask
        mask = torch.tensor(np.array(image)[0] > 0, dtype=torch.float32)

        return image, mask.unsqueeze(0)


# In[ ]:


# Model Definition (U-Net)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


# In[36]:


# Training
dataset = SyntheticShapesDataset(num_samples=1000)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, masks) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

