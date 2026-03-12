import torch
import torch.nn as nn
import torch.nn.functional as F

class ASLTranslatorCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLTranslatorCNN, self).__init__()
        
        # --- THE MAGNIFYING GLASSES (Convolutions) ---
        # Input: 3 color channels. Output: 16 feature maps.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        
        # Input: 16 feature maps. Output: 32 feature maps.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # The Pooling Layer (Shrinks the image size by half)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- THE JUDGE (Fully Connected Layers) ---
        # After two pools, our 64x64 image shrinks to 16x16. 
        # 32 channels * 16 height * 16 width = 8192 numbers to flatten!
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        
        # Final output layer: 128 neurons down to our 29 classes (A-Z, space, del, nothing)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. Pass through Conv1, apply ReLU (activation), then shrink with Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # 2. Pass through Conv2, apply ReLU, then shrink with Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # 3. Flatten the 3D tensor into a 1D line of numbers for the Judge
        x = torch.flatten(x, 1)
        
        # 4. Pass through the first Linear layer with ReLU
        x = F.relu(self.fc1(x))
        
        # 5. Output the final 29 predictions
        x = self.fc2(x)
        
        return x

# --- QUICK TEST TO VERIFY THE MATH ---
if __name__ == "__main__":
    # Create a dummy image batch (32 images, 3 channels, 64x64 pixels)
    dummy_data = torch.randn(32, 3, 64, 64)
    
    # Initialize our model
    model = ASLTranslatorCNN(num_classes=29)
    
    # Pass the dummy data through the model
    predictions = model(dummy_data)
    
    print(f"Input shape: {dummy_data.shape}")
    print(f"Output shape: {predictions.shape}")
    print("If output is [32, 29], the neural network math is perfectly aligned!")