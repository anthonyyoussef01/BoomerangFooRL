from ultralytics import YOLO
import os
import torch
import multiprocessing

if __name__ == '__main__':
    # Check if CUDA is available
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    # Load a pretrained YOLOv9c model
    multiprocessing.freeze_support()
    model = YOLO('yolov9c.pt')

    # Set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load the Boomerang Foo dataset from roboflow
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_yaml_path = os.path.join(current_dir, '..', '..', '..', 'data', 'game dataset', 'data.yaml')

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=100,     # Number of epochs
        imgsz=640,      # Image size
        batch=16,       # Batch size
        name='boomerang_foo_yolov9c'
    )

    # Save the trained model
    model.save('boomerang_foo_yolov9c.pt')
