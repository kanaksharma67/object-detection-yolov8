from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')  # Path to your trained weights

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Display output
    cv2.imshow('YOLO Detection', annotated_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#to run in terminal the yolo model after creating virtual environment just run this code below and a runs folder will be created with detect train weights folder in it 
# yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640


#to train model using image
# from ultralytics import YOLO
# model = YOLO('runs/detect/train/weights/best.pt')
# results = model('test_image.jpg')

# results[0].show()

