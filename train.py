from models import YOLO

# import pdb

# pdb.set_trace()

# Load a model
# model = YOLO('c2f-r2nfn-adown-slimc.yaml')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8.yaml')
model = YOLO('v8-2-adown-slimc.yaml')

# Train the model with 2 GPUs
results = model.train(data='data_2.yaml', epochs=10, imgsz=320, device='mps', batch=16, fraction=0.1)
