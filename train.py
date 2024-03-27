from models import YOLO

# import pdb

# pdb.set_trace()

# Load a model
model = YOLO('c2f-r2nfn.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.pt')

# Train the model with 2 GPUs
results = model.train(data='data.yaml', epochs=100, imgsz=320, device='mps', batch=16, fraction=1.0)
