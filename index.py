import cv2
import numpy as np

# Paths to your files
weights_path = "yolov4-tiny.weights"
config_path = "yolov4.cfg"
labels_path = "coco.names"
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)

# Load COCO class labels
with open(labels_path, "r") as f:
    classes = f.read().strip().split("\n")


face_cascade = cv2.CascadeClassifier(face_cascade_path)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break


    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)


    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Lowered confidence threshold for testing
                label = classes[class_id]

                # Process only for sunglasses and other objects
                if label == "sunglasses" or label != "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates for the detected object
                    x1 = int(center_x - w / 2)
                    y1 = int(center_y - h / 2)

                    boxes.append([x1, y1, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Non-max suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        indexes = indexes.flatten()

        
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

         
            glow_color = (0, 255, 0)  # Green color for highlight
            thickness = 3  # Border thickness

  
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), glow_color, thickness)

            
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        face_color = (255, 0, 0)  # Blue color for face
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
    
        cv2.putText(frame, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  
    cv2.imshow("Real-Time Object and Face Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
