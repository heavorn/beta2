from models.detect import YOLO
from PIL import Image

# Load a model
model = YOLO('v2_msfnns_300.pt')  # load a best weight
# model = YOLO('yolov8n.pt')  # load a pretrained model

img = "v22"

# Open the image file
image = Image.open(f"img/{img}.jpg")  # Replace with your image file path

# Resize the image to 640x640 pixels
# resized_image = image.resize((640, 640))


# import pdb
# pdb.set_trace()
# Train the model with 2 GPUs
results = model.predict(save=True, device='mps', source=image)

# print(results[0].boxes)

# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs




