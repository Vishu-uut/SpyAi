import cv2
import time
import numpy as np
import os
import threading
from collections import deque
from queue import Queue
import threading
import torch
import timm
from torchvision import transforms
from PIL import Image
import os

def frames_to_video(frames, output_video="output.mp4", fps=10):
    """
    Convert frame(s) to video file.
    
    Args:
        frames: Single frame (numpy array) or list of frames
        output_video (str): Output video file path
        fps (int): Frames per second
    
    Returns:
        str: Path to output video file if successful, None otherwise
    """
    # Convert single frame to list
    if isinstance(frames, np.ndarray) and len(frames.shape) == 3:
        frames = [frames]
    
    # Validate input
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        print("No valid frames provided!")
        return None

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

    video.release()
    return output_video


class ViTClassifier:
    def __init__(self, model_name='vit_base_patch16_224', num_classes=2, weights_path=None):
        self.device = torch.device("cuda:0")
        
        # Initialize model
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        
        # Load weights if provided
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing transforms
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """Process a single image for inference"""
        # image = Image.open(image_path).convert("RGB") # Removed image loading from path
        return self.transforms(image).unsqueeze(0).to(self.device)

    def classify_image(self, image, class_names=None):
        """
        Classify a single image
        
        Args:
            image (PIL.Image.Image): PIL Image object
            class_names (list, optional): List of class names
            
        Returns:
            dict: Classification results
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
        class_idx = predicted.item()
        class_probability = probabilities[0][class_idx].item()
        
        result = {
            'class_index': class_idx,
            'probability': class_probability,
            'all_probabilities': probabilities[0].tolist()
        }
        
        if class_names and class_idx < len(class_names):
            result['class_name'] = class_names[class_idx]
            
        return result

    def classify_batch(self, images, class_names=None):
        """
        Classify multiple images at once
        
        Args:
            image_paths (list): List of PIL Image objects
            class_names (list, optional): List of class names
            
        Returns:
            list: List of classification results
        """
        # Prepare batch
        batch_tensors = []
        for image in images:
            batch_tensors.append(self.preprocess_image(image))
        
        batch_tensor = torch.cat(batch_tensors, dim=0)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Process results
        results = []
        for idx in range(len(images)):
            class_idx = predicted[idx].item()
            class_probability = probabilities[idx][class_idx].item()
            
            result = {
                'class_index': class_idx,
                'probability': class_probability,
                'all_probabilities': probabilities[idx].tolist()
            }
            
            if class_names and class_idx < len(class_names):
                result['class_name'] = class_names[class_idx]
                
            results.append(result)
            
        return results


class ClipAnomalyDetector:
    def __init__(self, video_source=0, clip_duration=10, motion_threshold=0.5):
        self.video_source = video_source
        self.clip_duration = clip_duration
        self.motion_threshold = motion_threshold
        
        # Initialize the classifier
        self.classifier = ViTClassifier(
            model_name='vit_base_patch16_224', 
            num_classes=2, 
            weights_path=r"C:\Users\Dell\Desktop\backkk\ml_model\model.pth",
            #weights_only=True
        )
        
        self.output_dir = r"C:\Users\Dell\Desktop\backkk\backend\videos"
        os.makedirs(self.output_dir, exist_ok=True)

    def capture_clip(self):
        """Capture a 10-second video clip"""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source {self.video_source}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate total frames needed
        total_frames = self.clip_duration * fps
        frames = []
        
        print(f"Recording {self.clip_duration} second clip...")
        start_time = time.time()
        
        while len(frames) < total_frames and time.time() - start_time < self.clip_duration + 1:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            
            # Display recording progress
            cv2.putText(frame, f"Recording: {len(frames)}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Recording Clip", frame)
            cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        
        return frames, fps

    def extract_keyframes(self, frames):
        """Extract keyframes based on motion detection"""
        keyframes = []
        
        if not frames:
            return keyframes
            
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        
        for i, frame in enumerate(frames[1:], 1):
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute frame difference
            frame_diff = cv2.absdiff(prev_frame, curr_frame)
            non_zero_count = np.count_nonzero(frame_diff)
            
            # If motion detected, add to keyframes
            if non_zero_count > self.motion_threshold * frame_diff.size:
                resized_frame = cv2.resize(frame, (224, 224))
                keyframes.append({
                    "frame": resized_frame,
                    "frame_number": i,
                    "timestamp": i / len(frames) * self.clip_duration
                })
            
            prev_frame = curr_frame
            
        return keyframes

    def model_inference(self, frame):
        """Wrapper for classifier inference"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        result = self.classifier.classify_image(pil_image)
        return 1 if result['class_index'] == 1 else 0

    def process_clip(self):
        """Capture and process a clip"""
        try:
            # Capture clip
            frames, fps = self.capture_clip()
            if not frames:
                print("No frames captured")
                return
                
            print(f"Captured {len(frames)} frames")
            
            # Extract keyframes
            keyframes = self.extract_keyframes(frames)
            print(f"Extracted {len(keyframes)} keyframes")
            
            if not keyframes:
                print("No keyframes detected")
                return False
            
            # Find activity segments using the new logic
            segments = find_activity_segments(
                keyframes, 
                self.model_inference,
                tau_initial=2,
                beta=0.8,
                alpha=1
            )
            
            print(f"Detected {len(segments)} activity segments")
            
            # Save clips for each detected segment
            for i, (start, end) in enumerate(segments):
                # Extract frames for this segment
                segment_frames = []
                for frame_num in range(start, end + 1):
                    if 0 <= frame_num < len(frames):
                        segment_frames.append(frames[frame_num])
                
                if segment_frames:
                    # Save segment as video
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(
                        self.output_dir, 
                        f"clip_{timestamp}_segment_{i+1}.mp4"
                    )
                    frames_to_video(segment_frames, output_path, fps)
                    print(f"Saved segment {i+1} to {output_path}")
            
            return len(segments) > 0
            
        except Exception as e:
            print(f"Error processing clip: {e}")
            return False

def record_and_process_clips(duration_minutes=5):
    """Record and process clips for a specified duration"""
    detector = ClipAnomalyDetector(video_source=0)
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            print("\nStarting new clip recording...")
            detector.process_clip()
            
            # Brief pause between clips
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        cv2.destroyAllWindows()

def find_activity_segments(keyframes, model_inference, tau_initial=2, beta=0.8, alpha=1):
    """Identifies activity segments in a sequence of keyframes"""
    tau = tau_initial
    consecutive_frames = 0
    length = 0
    start = None
    segments = []

    for i, frame in enumerate(keyframes):
        prediction = model_inference(frame["frame"])
        
        if prediction == 1:
            if start is None:
                # Find the second previous keyframe if available
                if i >= 2:
                    start = keyframes[i-2]["frame_number"]
                elif i == 1:
                    start = keyframes[i-1]["frame_number"]
                else:
                    start = frame["frame_number"]
            length += 1
            consecutive_frames = 0
        
        elif prediction == 0:
            if start is not None:
                if consecutive_frames < tau:
                    consecutive_frames += 1
                    length += 1
                else:
                    # Find the next keyframe if available
                    if i < len(keyframes) - 1:
                        end = keyframes[i+1]["frame_number"]
                    else:
                        end = frame["frame_number"]
                    segments.append((start, end))
                    start = None
                    length = 0
                    consecutive_frames = 0
        
        tau = max(int(np.floor(beta * (length ** alpha))), 2)

    if start is not None:
        segments.append((start, keyframes[-1]["frame_number"]))

    return segments

# Record and process clips for 5 minutes
detector = ClipAnomalyDetector(
    video_source=1,
    clip_duration=120,  # 10-second clips
    motion_threshold=0.5  # Adjust as needed
)

# Process a single clip
detector.process_clip()

# Or record continuously
record_and_process_clips(duration_minutes=1)