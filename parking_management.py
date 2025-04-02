# Main application class
import cv2
import pickle
import numpy as np
import os
import threading
import time
import torch
from datetime import datetime
from tkinter import Tk, Label, Button, Frame, Canvas, Text, Scrollbar, OptionMenu, StringVar, IntVar, BooleanVar, \
    messagebox, ttk, BOTH, \
    TOP, LEFT, RIGHT, BOTTOM, X, Y, VERTICAL, HORIZONTAL, Toplevel
from PIL import Image, ImageTk

# Import our modules
from utils.gpu_utils import check_gpu_availability, gpu_adaptive_threshold, gpu_resize, diagnose_gpu
from utils.file_utils import ensure_directories_exist, load_parking_positions, save_parking_positions, save_log, \
    export_statistics
from detection.vehicle_detector import VehicleDetector
from detection.parking_detection import process_parking_frame, check_parking_space
from detection.vehicle_counting import detect_vehicles_traditional, detect_vehicles_ml, get_centroid

# Import UI modules
from ui.detection_tab import setup_detection_tab
from ui.setup_tab import setup_setup_tab
from ui.log_tab import setup_log_tab
from ui.stats_tab import setup_stats_tab
from ui.reference_tab import setup_reference_tab


class ParkingManagementSystem:
    DEFAULT_CONFIDENCE = 0.6
    DEFAULT_THRESHOLD = 500
    MIN_CONTOUR_SIZE = 40
    DEFAULT_OFFSET = 10
    DEFAULT_LINE_HEIGHT = 400

    def __init__(self, master):
        self.master = master
        self.master.title("Smart Parking Management System")
        self.master.geometry("1280x720")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initialize class variables
        self.running = False
        self.posList = []
        self.video_capture = None
        self.current_video = None
        self.vehicle_counter = 0
        self.matches = []  # For vehicle counting
        self.line_height = 400  # Default line height for vehicle detection
        self.min_contour_width = 40
        self.min_contour_height = 40
        self.offset = 10
        self.parking_threshold = 500  # Default threshold for parking space detection
        self.detection_mode = "parking"  # Default detection mode
        self.log_data = []  # For logging events

        # ML detection settings
        self.use_ml_detection = False
        self.ml_detector = None
        self.ml_confidence = self.DEFAULT_CONFIDENCE
        self.parking_threshold = self.DEFAULT_THRESHOLD
        self.min_contour_width = self.MIN_CONTOUR_SIZE
        self.min_contour_height = self.MIN_CONTOUR_SIZE
        self.offset = self.DEFAULT_OFFSET
        self.line_height = self.DEFAULT_LINE_HEIGHT

        # Thread safety
        self._cleanup_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.video_lock = threading.Lock()

        # GPU availability
        self.torch_gpu_available, self.cv_gpu_available, _ = check_gpu_availability()

        # Video reference map and dimensions
        self.video_reference_map = {
            "sample5.mp4": "saming1.png",
            "Video.mp4": "videoImg.png",
            "carPark.mp4": "carParkImg.png",
            "0": "webcamImg.png",  # Default for webcam
            "newVideo1.mp4": "newRefImage1.png",
            "newVideo2.mp4": "newRefImage2.png"
        }

        # Reference dimensions
        self.reference_dimensions = {
            "carParkImg.png": (1280, 720),
            "videoImg.png": (1280, 720),
            "webcamImg.png": (640, 480),
            "newRefImage1.png": (1280, 720),
            "newRefImage2.png": (1920, 1080)
        }
        self.current_reference_image = "carParkImg.png"  # Default

        # Load resources
        self.config_dir = "config"
        self.log_dir = "logs"
        self.ensure_directories_exist()
        self.load_parking_positions()

        # Setup UI components
        self.setup_ui()

        # Start a monitoring thread to log data
        self.monitor_thread = threading.Thread(target=self.monitoring_thread, daemon=True)
        self.monitor_thread.start()

        # Diagnose GPU
        self.diagnose_gpu()

    def ensure_directories_exist(self):
        """Ensure necessary directories exist"""
        directories = [self.config_dir, self.log_dir]
        ensure_directories_exist(directories)

    def __del__(self):
        self.cleanup_resources()

    def cleanup_resources(self):
        with self._cleanup_lock:
            if hasattr(self, 'video_capture') and self.video_capture:
                self.video_capture.release()
            if hasattr(self, 'ml_detector') and self.ml_detector:
                del self.ml_detector
            torch.cuda.empty_cache()

    def load_parking_positions(self, reference_image=None):
        if reference_image is None:
            reference_image = self.current_reference_image

        positions = load_parking_positions(self.config_dir, reference_image, self.log_event)
        self.posList = positions

        # Update counters
        self.total_spaces = len(self.posList)
        self.free_spaces = 0
        self.occupied_spaces = self.total_spaces

    def setup_ui(self):
        """Set up the application's user interface"""
        # Create main container
        self.main_container = ttk.Notebook(self.master)
        self.main_container.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        self.detection_tab = Frame(self.main_container)
        self.setup_tab = Frame(self.main_container)
        self.log_tab = Frame(self.main_container)
        self.stats_tab = Frame(self.main_container)
        self.reference_tab = Frame(self.main_container)

        self.main_container.add(self.detection_tab, text="Detection")
        self.main_container.add(self.setup_tab, text="Setup")
        self.main_container.add(self.log_tab, text="Logs")
        self.main_container.add(self.stats_tab, text="Statistics")
        self.main_container.add(self.reference_tab, text="References")

        # Setup each tab
        setup_detection_tab(self)
        setup_setup_tab(self)
        setup_log_tab(self)
        setup_stats_tab(self)
        setup_reference_tab(self)

    # The rest of the methods would be implemented here, but for clarity
    # only a subset are shown in this example. Each method would be moved
    # to the appropriate module.

    def update_status_info(self):
        """Update the status information display"""
        if hasattr(self, 'status_info'):
            status_text = f"Total Spaces: {self.total_spaces}\n"
            status_text += f"Free Spaces: {self.free_spaces}\n"
            status_text += f"Occupied: {self.occupied_spaces}\n"
            status_text += f"Vehicles Counted: {self.vehicle_counter}"

            # Update the status info label
            self.status_info.config(text=status_text)

    def log_event(self, message):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"

        # Add to log data
        self.log_data.append(log_entry)

        # Update log display if it exists
        if hasattr(self, 'log_text'):
            self.log_text.config(state="normal")
            self.log_text.insert("end", log_entry + "\n")
            self.log_text.see("end")  # Auto-scroll to the end
            self.log_text.config(state="disabled")

    def diagnose_gpu(self):
        """Run GPU diagnostics and log results"""
        results = diagnose_gpu()
        for result in results:
            self.log_event(result)

    def on_closing(self):
        """Handle window closing event"""
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.running = False
            if self.video_capture is not None:
                self.video_capture.release()
            self.master.destroy()

    def monitoring_thread(self):
        """Background thread for monitoring and periodic logging"""
        while True:
            # Record stats every hour if detection is running
            if self.running:
                self.record_current_stats()

            # Sleep for an hour (3600 seconds)
            time.sleep(3600)

    # Add any remaining methods you need...