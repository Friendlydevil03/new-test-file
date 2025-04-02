"""
Statistics Tab for Smart Parking Management System
Displays and manages parking statistics
"""

from tkinter import Label, Button, Frame, ttk
from tkinter import RIGHT, LEFT, BOTH, X, Y


class StatsTab:
    """Statistics tab UI class"""

    def __init__(self, parent, controller):
        """
        Initialize the statistics tab

        Args:
            parent: The parent frame
            controller: The main application controller
        """
        self.controller = controller
        self.parent = parent

        # Set up statistics tab components
        self._setup_stats_panel()

    def _setup_stats_panel(self):
        """Set up the statistics panel with treeview and controls"""
        # Stats tab frame
        self.stats_frame = Frame(self.parent, padx=10, pady=10)
        self.stats_frame.pack(fill=BOTH, expand=True)

        # Title
        Label(self.stats_frame, text="Parking Statistics", font=("Arial", 16, "bold")).pack(pady=10)

        # Statistics data
        self.stats_data_frame = Frame(self.stats_frame)
        self.stats_data_frame.pack(fill=BOTH, expand=True, pady=10)

        # Create Treeview for statistics
        self.stats_tree = ttk.Treeview(self.stats_data_frame,
                                       columns=("timestamp", "total", "free", "occupied", "vehicles"))

        # Define column headings
        self.stats_tree.heading("#0", text="")
        self.stats_tree.heading("timestamp", text="Timestamp")
        self.stats_tree.heading("total", text="Total Spaces")
        self.stats_tree.heading("free", text="Free Spaces")
        self.stats_tree.heading("occupied", text="Occupied Spaces")
        self.stats_tree.heading("vehicles", text="Vehicles Counted")

        # Define column widths
        self.stats_tree.column("#0", width=0, stretch=False)
        self.stats_tree.column("timestamp", width=200)
        self.stats_tree.column("total", width=100)
        self.stats_tree.column("free", width=100)
        self.stats_tree.column("occupied", width=100)
        self.stats_tree.column("vehicles", width=120)

        # Add scrollbar to treeview
        self.stats_vsb = ttk.Scrollbar(self.stats_data_frame, orient="vertical", command=self.stats_tree.yview)
        self.stats_tree.configure(yscrollcommand=self.stats_vsb.set)
        self.stats_vsb.pack(side=RIGHT, fill=Y)
        self.stats_tree.pack(side=LEFT, fill=BOTH, expand=True)

        # Statistics controls
        self.stats_control_frame = Frame(self.stats_frame)
        self.stats_control_frame.pack(fill=X, pady=10)

        Button(self.stats_control_frame, text="Clear Statistics",
               command=self.controller.clear_statistics).pack(side=RIGHT, padx=5)
        Button(self.stats_control_frame, text="Export Statistics",
               command=self.controller.export_statistics).pack(side=RIGHT, padx=5)
        Button(self.stats_control_frame, text="Record Current Stats",
               command=self.controller.record_current_stats).pack(side=RIGHT, padx=5)