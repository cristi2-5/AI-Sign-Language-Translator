import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the Transforms (The Preprocessing Pipeline)
# This resizes every image to 64x64 pixels and converts it to a 3D Tensor (numbers between 0 and 1)
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 2. Load the Dataset
# ImageFolder automatically looks at your folder names (A, B, C) and uses them as the labels
dataset_path = 'dataset/asl_alphabet_train/asl_alphabet_train' 
print(f"Loading images from: {dataset_path}...")

asl_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)

print(f"Total images loaded: {len(asl_dataset)}")
print(f"Classes found: {asl_dataset.classes}")

# 3. Create the DataLoader
# This groups the images into batches of 32 and shuffles them so the AI doesn't memorize the order
batch_size = 32
dataloader = DataLoader(asl_dataset, batch_size=batch_size, shuffle=True)

# 4. Test the Pipeline: Grab one batch of data
images, labels = next(iter(dataloader))

# Print the mathematical shape of the batch
print(f"\nShape of one batch of images: {images.shape}")
print(f"Shape of one batch of labels: {labels.shape}")

print("\nData pipeline is working perfectly! Ready for training.")