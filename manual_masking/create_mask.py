import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

class ROISelector:
    def __init__(self, ax, img):
        self.ax = ax
        self.img = img
        self.selector = PolygonSelector(ax, self.onselect)
        self.points = []

    def onselect(self, verts):
        self.points = verts
        self.selector.disconnect_events()
        plt.close()

    def get_mask(self):
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        if self.points:
            points = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

def create_mask_from_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    ax.imshow(img_rgb)
    ax.set_title("Draw a polygon around the area and double-click to finish.")
    roi_selector = ROISelector(ax, img_rgb)
    plt.show()
    return roi_selector.get_mask()

def apply_mask_to_video(video_path, output_video_path, mask, target_size=(640, 480)):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        resized_frame = cv2.resize(masked_frame, target_size)
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def main():

    # Set the video path
    video_path = 'input_video.asf'

    # Set the path for the processed video
    output_video_path = 'processed_video.asf'
    first_frame = get_first_frame(video_path)
    if first_frame is None:
        print("Error: Could not open video or read the first frame.")
        return

    mask = create_mask_from_image(first_frame)
    if mask is None:
        print("No mask created, exiting.")
        return

    apply_mask_to_video(video_path, output_video_path, mask, target_size=(640, 480))
    print("Processing complete.")

if __name__ == "__main__":
    main()
