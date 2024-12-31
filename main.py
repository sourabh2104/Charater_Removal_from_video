import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os

# Create necessary folders
for folder in ['input', 'output', 'models']:
    os.makedirs(folder, exist_ok=True)

class VideoObjectRemover:
    def __init__(self):
        print("Initializing Video Object Remover...")
        model_path = os.path.join('models', 'yolov8n.pt')
        
        # Download model if doesn't exist
        if not os.path.exists(model_path):
            print("Downloading AI model (this will happen only once)...")
            self.model = YOLO('yolov8n.pt')
            self.model.export()
            os.makedirs('models', exist_ok=True)
            # Save model for future use
            import shutil
            shutil.copy('yolov8n.pt', model_path)
        else:
            self.model = YOLO(model_path)
        
        print("Initialization complete!")

    def list_available_objects(self):
        """Show what objects can be removed"""
        print("\nCommon objects that can be removed:")
        common_objects = [
            'person', 'car', 'dog', 'cat', 'chair', 
            'bottle', 'laptop', 'tv', 'phone'
        ]
        for obj in common_objects:
            print(f"- {obj}")

    def process_video(self):
        """Main video processing function"""
        # Setup file selection dialog
        root = tk.Tk()
        root.withdraw()

        # Get input video
        print("\nSelect your input video...")
        input_path = filedialog.askopenfilename(
            title="Choose input video",
            initialdir="input",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if not input_path:
            print("No input video selected. Exiting...")
            return

        # Copy video to input folder if it's not already there
        if not input_path.startswith(os.path.join(os.getcwd(), 'input')):
            import shutil
            input_filename = os.path.basename(input_path)
            new_input_path = os.path.join('input', input_filename)
            shutil.copy(input_path, new_input_path)
            input_path = new_input_path
            print(f"Copied video to input folder: {input_filename}")

        # Get object to remove
        self.list_available_objects()
        target_object = input("\nWhat would you like to remove? ").lower()

        # Get output path
        output_filename = f"removed_{target_object}_" + os.path.basename(input_path)
        output_path = os.path.join('output', output_filename)

        # Process video
        print(f"\nProcessing video to remove {target_object}...")
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and remove objects
            results = self.model(frame)
            mask = np.zeros((height, width), dtype=np.uint8)

            for result in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = result
                class_name = self.model.names[int(cls)]
                
                if class_name == target_object:
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    mask[y1:y2, x1:x2] = 255

            # Remove object using inpainting
            if np.any(mask):
                frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

            out.write(frame)

            # Show progress
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}%", end="")

        # Cleanup
        cap.release()
        out.release()
        print(f"\n\nDone! Processed video saved to: {output_path}")

def main():
    print("=== Video Object Remover ===")
    remover = VideoObjectRemover()
    remover.process_video()

if __name__ == "__main__":
    main()