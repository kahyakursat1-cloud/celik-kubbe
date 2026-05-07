import torch
torch.cuda.empty_cache()

from ultralytics import YOLO

model = YOLO('runs/detect/runs/celikkubbe/yolo11m_military_v2/weights/best.pt')
model.train(
    data='celikkubbe_wsl.yaml',
    epochs=20,
    lr0=0.0001,
    warmup_epochs=0,
    batch=8,
    workers=4,
    device=0,
    project='runs/detect/training/celikkubbe/runs',
    name='celikkubbe_wsl2',
    save_period=5,
)
