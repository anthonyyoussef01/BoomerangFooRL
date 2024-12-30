from ultralytics import YOLO
import torch
import cv2

def adjust_label_position(existing_labels, x1, y1, label_height, image_height):
    for (lx, ly, lh) in existing_labels:
        if abs(lx - x1) < 50 and abs(ly - y1) < lh:
            y1 += lh + 10
    y1 = max(10, min(y1, image_height - 10))  # Ensure y1 stays within image boundaries
    return y1

if __name__ == '__main__':
    # Load the trained model
    model = YOLO('boomerang_foo_yolov9c.pt')

    # Set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and preprocess the image
    image_path = 'C:\\Users\\antho\\Pictures\\Screenshots\\Screenshot 2024-12-27 130819.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Perform inference
    results = model(image_rgb)

    # Visualize the labels on the image
    existing_labels = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        class_indices = result.boxes.cls.cpu().numpy().astype(int)  # Extract class indices
        class_names = [result.names[i] for i in class_indices]  # Map indices to class names
        for box, label in zip(boxes, class_names):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 150, 150), 3)
            label_y = adjust_label_position(existing_labels, x1, y1 - 10, 20, image_height)
            cv2.putText(image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 150, 150), 3)
            existing_labels.append((x1, label_y, 20))

    # Resize the window and display the image with labels
    window_name = 'Boomerang Foo Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # Set the desired window size
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()