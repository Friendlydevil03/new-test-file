"""
Log Tab for Smart Parking Management System
Displays and manages system logs
"""

from tkinter import Text, Label, Button, Frame, Scrollbar
from tkinter import LEFT, RIGHT, BOTH, X, Y


class LogTab:
    """Log tab UI class for system logs"""

    def __init__(self, parent, controller):
        """
        Initialize the log tab

        Args:
            parent: The parent frame
            controller: The main application controller
        """
        self.controller = controller
        self.parent = parent

        # Set up log tab components
        self._setup_log_panel()

    def _setup_log_panel(self):
        """Set up the log panel with text display and controls"""
        # Log tab frame
        self.log_frame = Frame(self.parent, padx=10, pady=10)
        self.log_frame.pack(fill=BOTH, expand=True)

        # Title and controls
        self.log_header = Frame(self.log_frame)
        self.log_header.pack(fill=X, pady=5)

        Label(self.log_header, text="System Logs", font=("Arial", 14, "bold")).pack(side=LEFT)

        self.clear_log_button = Button(self.log_header, text="Clear Log", command=self.controller.clear_log)
        self.clear_log_button.pack(side=RIGHT, padx=5)

        self.save_log_button = Button(self.log_header, text="Save Log", command=self.controller.save_log)
        self.save_log_button.pack(side=RIGHT, padx=5)

        # Log text area with scrollbar
        self.log_text_frame = Frame(self.log_frame)
        self.log_text_frame.pack(fill=BOTH, expand=True, pady=10)

        self.log_text = Text(self.log_text_frame, wrap="word", height=20)
        self.log_text.pack(side=LEFT, fill=BOTH, expand=True)

        self.log_scrollbar = Scrollbar(self.log_text_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=RIGHT, fill=Y)

        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        self.log_text.config(state="disabled")

    def append_log(self, text):
        """
        Append text to the log display

        Args:
            text: The text to append
        """
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")  # Auto-scroll to the end
        self.log_text.config(state="disabled")

    def clear_display(self):
        """Clear the log text display"""
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, "end")
        self.log_text.config(state="disabled")