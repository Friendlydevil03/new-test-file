"""
Setup Tab for Smart Parking Management System
Allows users to define and configure parking spaces
"""

from tkinter import Canvas, Label, Button, Frame, StringVar
from tkinter import TOP, LEFT, RIGHT, X, Y, BOTH


class SetupTab:
    """Setup tab UI class for parking space configuration"""

    def __init__(self, parent, controller):
        """
        Initialize the setup tab

        Args:
            parent: The parent frame
            controller: The main application controller
        """
        self.controller = controller
        self.parent = parent

        # Setup UI components
        self._setup_control_frame()
        self._setup_canvas_frame()
        self._setup_mouse_events()

        # Load reference image
        self.controller.load_reference_image()

    def _setup_control_frame(self):
        """Set up the control frame with buttons and options"""
        # Frame for setup controls
        self.setup_control_frame = Frame(self.parent, padx=10, pady=10)
        self.setup_control_frame.pack(side=TOP, fill=X)

        Label(self.setup_control_frame, text="Parking Space Setup", font=("Arial", 14, "bold")).pack(side=LEFT, padx=10)

        self.setup_instructions = Label(self.setup_control_frame,
                                        text="Left-click and drag to draw spaces. Right-click to delete spaces.",
                                        font=("Arial", 10))
        self.setup_instructions.pack(side=LEFT, padx=10)

        # Calibration buttons
        calibration_frame = Frame(self.setup_control_frame)
        calibration_frame.pack(side=LEFT, padx=20)

        Label(calibration_frame, text="Calibration:").pack(side=LEFT)

        Button(calibration_frame, text="↑", command=lambda: self.controller.shift_all_spaces(0, -5)).pack(side=LEFT)
        Button(calibration_frame, text="↓", command=lambda: self.controller.shift_all_spaces(0, 5)).pack(side=LEFT)
        Button(calibration_frame, text="←", command=lambda: self.controller.shift_all_spaces(-5, 0)).pack(side=LEFT)
        Button(calibration_frame, text="→", command=lambda: self.controller.shift_all_spaces(5, 0)).pack(side=LEFT)

        # Reference image selection
        self.ref_image_var = StringVar(value=self.controller.current_reference_image)
        Label(self.setup_control_frame, text="Reference Image:").pack(side=LEFT, padx=10)

        self.ref_image_menu = OptionMenu(self.setup_control_frame, self.ref_image_var,
                                         *list(self.controller.video_reference_map.values()),
                                         command=self.controller.load_reference_image)
        self.ref_image_menu.pack(side=LEFT, padx=10)

        # Control buttons
        self.save_spaces_button = Button(self.setup_control_frame, text="Save Spaces",
                                         command=self.controller.save_parking_spaces)
        self.save_spaces_button.pack(side=RIGHT, padx=10)

        self.clear_spaces_button = Button(self.setup_control_frame, text="Clear All",
                                          command=self.controller.clear_all_spaces)
        self.clear_spaces_button.pack(side=RIGHT, padx=10)

        self.associate_button = Button(self.setup_control_frame, text="Associate Video",
                                       command=self.controller.associate_video_with_reference)
        self.associate_button.pack(side=RIGHT, padx=10)

    def _setup_canvas_frame(self):
        """Set up the canvas for drawing parking spaces"""
        # Frame for the setup canvas
        self.setup_canvas_frame = Frame(self.parent)
        self.setup_canvas_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.setup_canvas = Canvas(self.setup_canvas_frame, bg='black')
        self.setup_canvas.pack(fill=BOTH, expand=True)

    def _setup_mouse_events(self):
        """Set up mouse event bindings for drawing spaces"""
        # Setup mouse events
        self.drawing = False
        self.start_x, self.start_y = -1, -1
        self.current_rect = None

        self.setup_canvas.bind("<ButtonPress-1>", self.controller.on_mouse_down)
        self.setup_canvas.bind("<B1-Motion>", self.controller.on_mouse_move)
        self.setup_canvas.bind("<ButtonRelease-1>", self.controller.on_mouse_up)
        self.setup_canvas.bind("<ButtonPress-3>", self.controller.on_right_click)