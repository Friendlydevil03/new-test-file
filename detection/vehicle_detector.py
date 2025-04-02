import cv2
import pickle
import cvzone
import numpy as np
import os
import threading
import time
import torch
from torchvision.models import detection
from tkinter import Toplevel
from datetime import datetime
from tkinter import Tk, Label, Button, Frame, Canvas, Text, Scrollbar, OptionMenu, StringVar, IntVar, BooleanVar, messagebox, ttk, BOTH, \
    TOP, LEFT, RIGHT, BOTTOM, X, Y, VERTICAL, HORIZONTAL
from PIL import Image, ImageTk

class VehicleDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Track memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache first
            print(f"GPU Memory before model: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")

        try:
            # Load pre-trained model with updated API
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            self.model = detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.model.to(self.device)
            self.model.eval()

            # Verify model is on correct device
            print(f"Model device: {next(self.model.parameters()).device}")

            # Report memory usage after model load
            if torch.cuda.is_available():
                print(f"GPU Memory after model: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # COCO class names
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat'
        ]
        self.vehicle_classes = [2, 3, 5, 6, 7, 8]  # Indices of vehicle classes

        # Use smaller input size for inference (keeps aspect ratio)
        self.max_size = 480  # Lower this for more speed, raise for more accuracy

        # Warm up the model
        if torch.cuda.is_available():
            dummy_input = torch.zeros((1, 3, self.max_size, self.max_size), device=self.device)
            try:
                with torch.no_grad():
                    _ = self.model(dummy_input)
                print("Model warm-up completed")
            except Exception as e:
                print(f"Model warm-up failed: {str(e)}, continuing anyway")

    def detect_vehicles(self, frame):
        orig_h, orig_w = frame.shape[:2]

        # Resize to target size while maintaining aspect ratio
        scale = min(self.max_size / orig_h, self.max_size / orig_w)
        if scale < 1.0:
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        # Convert frame to tensor
        img = torch.from_numpy(frame.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            predictions = self.model(img)

        # Extract detections
        boxes = predictions[0]['boxes'].cpu().numpy().astype(int)
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        # Filter by confidence and vehicle classes
        vehicle_detections = []
        for box, score, label in zip(boxes, scores, labels):
            if score > self.confidence_threshold and label in self.vehicle_classes:
                x1, y1, x2, y2 = box

                # Scale back to original size if resized
                if scale < 1.0:
                    x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

                vehicle_detections.append((x1, y1, x2, y2, score, label))

        return vehicle_detections
