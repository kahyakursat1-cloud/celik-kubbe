import os
import yaml
from ultralytics import YOLO

def create_dataset_yaml(output_path, base_dir):
    """Generate the data.yaml file for YOLOv11."""
    data = {
        'path': base_dir,
        'train': 'train/images_aug', # Folder with augmented training images
        'val': 'val/images_aug',   # Folder with augmented validation images
        'names': {
            0: 'f16',
            1: 'uav',
            2: 'missile',
            3: 'helicopter'
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Dataset YAML created at {output_path}")

def train_model(data_yaml, epochs=50, imgsz=640):
    """Train the YOLOv11 model."""
    # Load previously highly-trained model for transfer learning
    pretrained_model = r"C:\Users\Victus\Desktop\dataset olusturma\runs\map95_training\weights\best.pt"
    print(f"Loading base model from {pretrained_model}...")
    model = YOLO(pretrained_model)
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0, # GPU
        name="rocket_multitask_run"
    )
    return results

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data", "synthetic")
    YAML_PATH = os.path.join(DATA_DIR, "data.yaml")
    
    # 1. Create YAML
    create_dataset_yaml(YAML_PATH, DATA_DIR)
    
    # 2. Start Training
    train_model(YAML_PATH, epochs=50, imgsz=960) # Use 960 for higher accuracy as per goal
