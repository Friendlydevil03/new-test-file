import torch
from tkinter import Frame, Label, Button, OptionMenu, StringVar, IntVar, BOTH, LEFT, RIGHT, X, Y
from tkinter import ttk, Canvas

from detection.vehicle_detector import VehicleDetector


class DetectionTab:
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app

        # Create the tab frame
        self.frame = Frame(parent)
        self.setup_ui()

        # Frame processing thread
        self.frame_processing_thread = None
        self.frame_skip = 3  # Process every 3rd frame with ML
        self.frame_count = 0

    def setup_ui(self):
        """Set up the detection tab UI with scrollable right panel"""
        # Main frame for detection
        self.detection_frame = Frame(self.frame)
        self.detection_frame.pack(fill=BOTH, expand=True)

        # Left side - Video feed
        self.video_frame = Frame(self.detection_frame)
        self.video_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.video_canvas = Canvas(self.video_frame, bg='black')
        self.video_canvas.pack(fill=BOTH, expand=True, padx=5, pady=5)

        # Right side - Controls and info with scrolling
        right_frame = Frame(self.detection_frame, width=300)
        right_frame.pack(side=RIGHT, fill=Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        # Add a canvas and scrollbar for scrolling
        control_canvas = Canvas(right_frame, width=280)
        control_canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=control_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)

        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.bind('<Configure>', lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))

        # Create the inner frame for controls
        self.control_frame = Frame(control_canvas)
        control_canvas_window = control_canvas.create_window((0, 0), window=self.control_frame, anchor="nw", width=280)

        # Mode selection
        Label(self.control_frame, text="Detection Mode:", font=("Arial", 12)).pack(pady=(10, 5))
        self.mode_var = StringVar(value="parking")
        self.mode_menu = OptionMenu(self.control_frame, self.mode_var,
                                    "parking", "vehicle",
                                    command=self.switch_detection_mode)
        self.mode_menu.pack(fill=X, pady=5)

        # ML detection toggle
        self.ml_frame = Frame(self.control_frame)
        self.ml_frame.pack(fill=X, pady=5)
        self.ml_var = StringVar(value="Off")
        Label(self.ml_frame, text="ML Detection:").pack(side=LEFT)
        self.ml_toggle = OptionMenu(self.ml_frame, self.ml_var, "Off", "On", command=self.toggle_ml_detection)
        self.ml_toggle.pack(side=LEFT, padx=5)

        # ML confidence slider
        Label(self.control_frame, text="ML Confidence:").pack(anchor="w")
        self.ml_confidence_scale = ttk.Scale(self.control_frame, from_=0.1, to=0.9,
                                             orient="horizontal", value=0.6,
                                             command=self.update_ml_confidence)
        self.ml_confidence_scale.pack(fill=X, pady=5)
        self.ml_confidence_label = Label(self.control_frame, text=f"Value: 0.6")
        self.ml_confidence_label.pack(anchor="w")

        # Video source selection
        Label(self.control_frame, text="Video Source:", font=("Arial", 12)).pack(pady=(10, 5))
        self.video_source_var = StringVar(value="sample5.mp4")
        self.video_sources = ["sample5.mp4", "Video.mp4", "0", "carPark.mp4", "newVideo1.mp4", "newVideo2.mp4"]
        self.video_menu = OptionMenu(self.control_frame, self.video_source_var,
                                     *self.video_sources,
                                     command=self.switch_video_source)
        self.video_menu.pack(fill=X, pady=5)

        # Status information
        Label(self.control_frame, text="Status Information", font=("Arial", 14, "bold")).pack(pady=10)
        self.status_info = Label(self.control_frame,
                                 text="Total Spaces: 0\nFree Spaces: 0\nOccupied: 0\nVehicles Counted: 0",
                                 font=("Arial", 12), justify=LEFT)
        self.status_info.pack(pady=5, fill=X)

        # Status indicator
        self.status_label = Label(self.control_frame, text="Status: Stopped", fg="red", font=("Arial", 12))
        self.status_label.pack(pady=5, fill=X)

        # Buttons
        self.button_frame = Frame(self.control_frame)
        self.button_frame.pack(fill=X, pady=10)

        self.start_button = Button(self.button_frame, text="Start Detection", command=self.start_detection)
        self.start_button.pack(fill=X, pady=5)

        self.stop_button = Button(self.button_frame, text="Stop Detection", command=self.stop_detection,
                                  state="disabled")
        self.stop_button.pack(fill=X, pady=5)

        self.reset_button = Button(self.button_frame, text="Reset Counters", command=self.reset_counters)
        self.reset_button.pack(fill=X, pady=5)

        # Advanced settings
        Label(self.control_frame, text="Settings", font=("Arial", 12, "bold")).pack(pady=(15, 5))

        # Threshold slider
        Label(self.control_frame, text="Detection Threshold:").pack(anchor="w")
        self.threshold_scale = ttk.Scale(self.control_frame, from_=100, to=1000,
                                         orient="horizontal", value=self.main_app.parking_threshold,
                                         command=self.update_threshold)
        self.threshold_scale.pack(fill=X, pady=5)
        self.threshold_label = Label(self.control_frame, text=f"Value: {self.main_app.parking_threshold}")
        self.threshold_label.pack(anchor="w")

        # Performance section
        Label(self.control_frame, text="Performance Settings", font=("Arial", 12, "bold")).pack(pady=(15, 5))

        # Frame skip control
        skip_frame = Frame(self.control_frame)
        skip_frame.pack(fill=X, pady=5)

        Label(skip_frame, text="ML Frame Skip:").pack(side=LEFT)
        self.frame_skip_var = IntVar(value=3)

        skip_options = [(1, "Every frame"), (2, "Every 2nd"), (3, "Every 3rd"), (5, "Every 5th"), (8, "Every 8th")]
        for val, text in skip_options:
            ttk.Radiobutton(skip_frame, text=text, variable=self.frame_skip_var, value=val,
                            command=lambda v=val: self.set_frame_skip_rate(v)).pack(anchor="w")

        # GPU status
        gpu_status = "Available" if torch.cuda.is_available() else "Not Available"
        gpu_color = "green" if torch.cuda.is_available() else "red"
        Label(self.control_frame, text=f"GPU: {gpu_status}", fg=gpu_color, font=("Arial", 10, "bold")).pack(pady=5)

        # GPU test button
        Button(self.control_frame, text="Test GPU", command=self.test_gpu).pack(pady=5)

        # Make sure the canvas can scroll to all controls
        self.control_frame.update_idletasks()  # Update geometry
        control_canvas.config(scrollregion=control_canvas.bbox("all"))

        # Define mousewheel scrolling function
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # Add mousewheel scrolling
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Store control_canvas reference for later use
        self.control_canvas = control_canvas

    def switch_detection_mode(self, mode):
        """Switch between parking and vehicle detection modes"""
        self.main_app.detection_mode = self.mode_var.get()
        self.main_app.log_event(f"Switched to {self.main_app.detection_mode} detection mode")

    def toggle_ml_detection(self, value):
        """Toggle ML-based detection on or off"""
        self.main_app.use_ml_detection = (value == "On")
        if self.main_app.use_ml_detection and not self.main_app.ml_detector:
            try:
                self.main_app.ml_detector = VehicleDetector(confidence_threshold=self.main_app.ml_confidence)
            except Exception as e:
                self.main_app.log_event(f"Failed to initialize ML detector: {str(e)}")
                self.ml_var.set("Off")
                self.main_app.use_ml_detection = False

        self.main_app.log_event(f"ML detection: {value}")

    def update_ml_confidence(self, value):
        """Update ML detection confidence threshold"""
        self.main_app.ml_confidence = float(value)
        self.ml_confidence_label.config(text=f"Value: {self.main_app.ml_confidence:.1f}")

        if self.main_app.ml_detector:
            self.main_app.ml_detector.confidence_threshold = self.main_app.ml_confidence

    def switch_video_source(self, source):
        """Switch to a different video source"""
        # Implementation details
        pass

    def start_detection(self):
        """Start the detection process"""
        # Implementation details
        pass

    def stop_detection(self):
        """Stop the detection process"""
        # Implementation details
        pass

    def reset_counters(self):
        """Reset vehicle and parking counters"""
        # Implementation details
        pass

    def update_threshold(self, value):
        """Update the parking detection threshold"""
        # Implementation details
        pass

    def set_frame_skip_rate(self, rate):
        """Set the frame skip rate for ML detection"""
        # Implementation details
        pass

    def test_gpu(self):
        """Test GPU performance and availability"""
        # Implementation details
        pass