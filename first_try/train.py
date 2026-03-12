import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import your model from the other file!
from first_try.model import ASLTranslatorCNN  # Ensure your model file is named model.py

# 1. Setup the Device (Use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Re-create the Data Loader (The Water)
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# Make sure this path matches the one we fixed earlier!
dataset_path = 'dataset/asl_alphabet_train/asl_alphabet_train' 
asl_dataset = datasets.ImageFolder(root=dataset_path, transform=data_transforms)
dataloader = DataLoader(asl_dataset, batch_size=32, shuffle=True)

# 3. Initialize the Model, the Judge, and the Plumber
model = ASLTranslatorCNN(num_classes=29).to(device)

# The Loss Function (Calculates how wrong the model's guess is)
criterion = nn.CrossEntropyLoss()

# The Optimizer (The Plumber: adjusts the valves to lower the loss)
# lr=0.001 is the "Learning Rate" - how drastically it turns the valves each time
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# 4. THE TRAINING LOOP
num_epochs = 2 # An "epoch" is one full pass through all 87,000 images
print("Starting the training process...")

for epoch in range(num_epochs):
    running_loss = 0.0
    
    # Loop through the dataset in batches of 32
    for i, (images, labels) in enumerate(dataloader):
        # Send data to the correct device (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Step A: Zero the gradients (Tell the plumber to wipe their memory from the last batch)
        optimizer.zero_grad()
        
        # Step B: Forward Pass (Make a guess)
        outputs = model(images)
        
        # Step C: Calculate the Loss (How wrong was the guess?)
        loss = criterion(outputs, labels)
        
        # Step D: Backpropagation (Calculate exactly which valves to turn)
        loss.backward()
        
        # Step E: Optimize (Actually turn the valves!)
        optimizer.step()
        
        # Keep track of the math to print it out
        running_loss += loss.item()
        
        # Print an update every 100 batches so we know it hasn't frozen
        if i % 100 == 99:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training Complete!")

# 5. Save the trained brain!
torch.save(model.state_dict(), "asl_model.pth")
print("Model saved to asl_model.pth. We can now use it in the webcam!")