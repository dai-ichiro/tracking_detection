import sys
import torch

fname = sys.argv[1]
pt_path = sys.argv[2]

model = torch.hub.load('ultralytics/yolov5', 'custom', path = pt_path)
results = model([fname])
results.show()