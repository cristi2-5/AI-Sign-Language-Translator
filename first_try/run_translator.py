import cv2
import torch
from torchvision import transforms
from PIL import Image
from first_try.model import ASLTranslatorCNN # Import your blueprint!

# 1. Setup the Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLTranslatorCNN(num_classes=29).to(device)

# Load the trained weights (the valves you just adjusted)
model.load_state_dict(torch.load("asl_model.pth", map_location=device))
model.eval() # THIS IS CRITICAL: Tells the model "Stop learning, start predicting"

# The exact list of classes from your dataset
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
           'del', 'nothing', 'space']

# 2. Setup the Preprocessing Pipeline
# We have to treat the webcam frame exactly like the training data!
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 3. Turn on the Webcam
cap = cv2.VideoCapture(0)
print("Webcam is live! Put your hand in the green box. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip the frame so it acts like a mirror
    frame = cv2.flip(frame, 1)
    
    # --- THE VIEWFINDER (Region of Interest) ---
    # Draw a 250x250 green square on the right side of the screen
    x1, y1, x2, y2 = 350, 100, 600, 350
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Crop the image to JUST the green square
    roi = frame[y1:y2, x1:x2]
    
    # --- TRANSLATION TIME ---
    # OpenCV uses BGR colors, but PyTorch trained on RGB colors. We must convert!
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # Convert the numpy array to a PIL Image, then apply our transforms
    pil_image = Image.fromarray(roi_rgb)
    tensor_image = transform(pil_image).unsqueeze(0).to(device) # unsqueeze(0) adds a fake batch size of 1
    
    # Pass the image through the brain!
    with torch.no_grad(): # Tells PyTorch not to calculate gradients (saves memory)
        prediction = model(tensor_image)
        
        # Find the highest probability
        predicted_index = torch.argmax(prediction, dim=1).item()
        predicted_letter = classes[predicted_index]
    
    # --- DISPLAY THE RESULT ---
    # Write the predicted letter on the screen in big blue text
    cv2.putText(frame, f"Prediction: {predicted_letter}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    
    # Show the final window
    cv2.imshow("ASL Translator", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()