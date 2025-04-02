#!/usr/bin/env python3
"""
Smart Parking Management System
Main entry point for the application
"""

import os
import sys
from tkinter import Tk

# Add modules to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main application class
from parking_management import ParkingManagementSystem


def main():
    """Main entry point for the application"""
    # Create the main Tkinter root window
    root = Tk()

    # Initialize the application
    app = ParkingManagementSystem(root)

    # Start the Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    main()