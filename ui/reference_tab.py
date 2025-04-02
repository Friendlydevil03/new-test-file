"""
Reference Tab for Smart Parking Management System
Manages reference images for parking space detection
"""

import os
import cv2
from tkinter import Label, Button, Frame, Canvas, ttk
from tkinter import LEFT, RIGHT, X, Y, BOTH
from PIL import Image, ImageTk

class ReferenceTab:
    """Reference tab UI class for managing reference images"""

    def __init__(self, parent, controller):
        """
        Initialize the reference tab

        Args:
            parent: The parent frame
            controller: The main application controller
        """
        self.controller = controller
        self.parent = parent

        # Set up reference tab components
        self._setup_reference_panel()

    def _setup_reference_panel(self):
        """Set up the reference panel with treeview and preview"""
        # Reference tab frame
        self.reference_frame = Frame(self.parent, padx=10, pady=10)
        self.reference_frame.pack(fill=BOTH, expand=True)

        # Header frame
        header_frame = Frame(self.reference_frame)
        header_frame.pack(fill=X, pady=5)

        Label(header_frame, text="Reference Images", font=("Arial", 14, "bold")).pack(side=LEFT)

        # Add buttons
        Button(header_frame, text="Add Reference",
              command=self.controller.browse_reference_image).pack(side=RIGHT, padx=5)
        Button(header_frame, text="Associate Video",
              command=self.controller.associate_video_with_reference).pack(side=RIGHT, padx=5)

        # Create Treeview for references
        ref_tree_frame = Frame(self.reference_frame)
        ref_tree_frame.pack(fill=BOTH, expand=True, pady=10)

        self.ref_tree = ttk.Treeview(ref_tree_frame, columns=("image", "dimensions", "associated_videos"))

        # Define column headings
        self.ref_tree.heading("#0", text="")
        self.ref_tree.heading("image", text="Reference Image")
        self.ref_tree.heading("dimensions", text="Dimensions")
        self.ref_tree.heading("associated_videos", text="Associated Videos")

        # Define column widths
        self.ref_tree.column("#0", width=0, stretch=False)
        self.ref_tree.column("image", width=200)
        self.ref_tree.column("dimensions", width=150)
        self.ref_tree.column("associated_videos", width=300)

        # Add scrollbar
        ref_vsb = ttk.Scrollbar(ref_tree_frame, orient="vertical", command=self.ref_tree.yview)
        self.ref_tree.configure(yscrollcommand=ref_vsb.set)
        ref_vsb.pack(side=RIGHT, fill=Y)
        self.ref_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Preview frame
        preview_frame = Frame(self.reference_frame)
        preview_frame.pack(fill=BOTH, expand=True, pady=10)

        Label(preview_frame, text="Image Preview", font=("Arial", 12, "bold")).pack(pady=5)

        self.preview_canvas = Canvas(preview_frame, bg="black", height=300)
        self.preview_canvas.pack(fill=BOTH, expand=True)

        # Populate the reference tree
        self.populate_reference_tree()

        # Bind selection event
        self.ref_tree.bind("<<TreeviewSelect>>", self.on_reference_select)

    def populate_reference_tree(self):
        """Populate the reference image tree with data"""
        # Clear existing items
        for item in self.ref_tree.get_children():
            self.ref_tree.delete(item)

        # Add each reference image
        for ref_img in set(self.controller.video_reference_map.values()):
            # Find associated videos
            associated = [vid for vid, img in self.controller.video_reference_map.items() if img == ref_img]
            associated_str = ", ".join(associated)

            # Get dimensions
            dimensions = self.controller.reference_dimensions.get(ref_img, "Unknown")
            if dimensions != "Unknown":
                dimensions_str = f"{dimensions[0]}x{dimensions[1]}"
            else:
                dimensions_str = "Unknown"

            # Insert into tree
            self.ref_tree.insert("", "end", values=(ref_img, dimensions_str, associated_str))

    def on_reference_select(self, event):
        """Handle reference image selection"""
        selection = self.ref_tree.selection()
        if selection:
            item = selection[0]
            ref_img = self.ref_tree.item(item, "values")[0]

            # Display the image in the preview canvas
            try:
                if os.path.exists(ref_img):
                    img = cv2.imread(ref_img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize for preview
                    preview_height = 300
                    ratio = preview_height / img.shape[0]
                    preview_width = int(img.shape[1] * ratio)

                    img = cv2.resize(img, (preview_width, preview_height))

                    # Convert to PhotoImage
                    img_pil = Image.fromarray(img)
                    img_tk = ImageTk.PhotoImage(image=img_pil)

                    # Update canvas
                    self.preview_canvas.config(width=preview_width, height=preview_height)
                    self.preview_canvas.create_image(0, 0, anchor="nw", image=img_tk)
                    self.preview_canvas.image = img_tk  # Keep a reference
            except Exception as e:
                self.controller.log_event(f"Error previewing reference image: {str(e)}")