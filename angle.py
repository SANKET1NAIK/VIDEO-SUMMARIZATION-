import torch
from ultralytics import YOLO

# Convert water model
water_model = YOLO('water.pt')
water_model.export(format='onnx', dynamic=True, simplify=True)

# Convert feeder model
feeder_model = YOLO('feeder.pt')
feeder_model.export(format='onnx', dynamic=True, simplify=True)
