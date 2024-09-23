import os
import shutil
import cv2
import numpy as np

class BedDetector:
    def __init__(self, yolo_weights, yolo_cfg, coco_names):
        self.net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
        with open(coco_names, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.classes[class_id] == "bed":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
        return boxes

def sample_frames(video_path, num_samples=100):

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(num_samples):
        frame_id = int((i + 1) * frame_count / (num_samples + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_bed_boxes(video_path, detector, num_samples=100):
    frames = sample_frames(video_path, num_samples)
    boxes = []
    for frame in frames:
        boxes.extend(detector.detect_objects(frame))
    if not boxes:
        return []
    
    return [boxes[0]]

def process_videos(input_folder, output_folder, detector, target_size=(640, 480), bitrate="500k"):
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".asf"):
                input_video = os.path.join(root, filename)

                relative_path = os.path.relpath(root, input_folder)
                output_dir = os.path.join(output_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                output_video = os.path.join(output_dir, filename)  # Use original filename

                boxes = get_bed_boxes(input_video, detector)
                if not boxes:
                    print(f"No bed detected in video: {filename}. Copying the original video.")
                    shutil.copy(input_video, output_video)
                    continue

                box = boxes[0]
                cap = cv2.VideoCapture(input_video)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                x = max(0, box[0])
                y = max(0, box[1])
                w = min(width - x, box[2])
                h = min(height - y, box[3])

                # Open video capture and writer
                cap = cv2.VideoCapture(input_video)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video, fourcc, fps, target_size, isColor=True)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Crop the frame to the detected bed area
                    cropped_frame = frame[y:y+h, x:x+w]
                    # Resize the cropped frame to the target size
                    resized_frame = cv2.resize(cropped_frame, target_size)
                    # Write the resized frame to the output video
                    out.write(resized_frame)

                cap.release()
                out.release()

                print(f"Video {filename} for {relative_path} has been processed successfully.")

def encode_video(input_path, output_path, bitrate):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
