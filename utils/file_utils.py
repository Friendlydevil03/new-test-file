import os
import pickle
from datetime import datetime


def ensure_directories_exist(directories):
    """Ensure necessary directories exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def load_parking_positions(config_dir, current_reference_image, log_event_callback=None):
    """Load parking positions from file"""
    try:
        pos_file = os.path.join(config_dir, f'CarParkPos_{os.path.splitext(current_reference_image)[0]}')

        if not os.path.exists(config_dir):
            raise FileNotFoundError(f"Config directory {config_dir} does not exist")

        if os.path.exists(pos_file):
            with open(pos_file, 'rb') as f:
                positions = pickle.load(f)
            return positions
        else:
            return []

    except FileNotFoundError as e:
        if log_event_callback:
            log_event_callback(f"Position file not found: {str(e)}")
        return []
    except PermissionError as e:
        if log_event_callback:
            log_event_callback(f"Permission denied: {str(e)}")
        return []
    except Exception as e:
        if log_event_callback:
            log_event_callback(f"Unexpected error: {str(e)}")
        return []


def save_parking_positions(config_dir, current_reference_image, positions, log_event_callback=None):
    """Save parking positions to file"""
    try:
        pos_file = os.path.join(config_dir, f'CarParkPos_{os.path.splitext(current_reference_image)[0]}')
        with open(pos_file, 'wb') as f:
            pickle.dump(positions, f)

        if log_event_callback:
            log_event_callback(f"Saved {len(positions)} parking spaces for {current_reference_image}")

        return True
    except Exception as e:
        if log_event_callback:
            log_event_callback(f"Failed to save parking spaces: {str(e)}")
        return False


def save_log(log_dir, log_data, log_event_callback=None):
    """Save log data to a file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_log_{timestamp}.txt")

        with open(filename, 'w') as f:
            for entry in log_data:
                f.write(entry + "\n")

        if log_event_callback:
            log_event_callback(f"Log saved to {filename}")

        return filename
    except Exception as e:
        if log_event_callback:
            log_event_callback(f"Failed to save log: {str(e)}")
        return None


def export_statistics(log_dir, stats_data, log_event_callback=None):
    """Export statistics to CSV file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(log_dir, f"parking_stats_{timestamp}.csv")

        with open(filename, 'w') as f:
            f.write("Timestamp,Total Spaces,Free Spaces,Occupied Spaces,Vehicles Counted\n")
            for row in stats_data:
                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

        if log_event_callback:
            log_event_callback(f"Statistics exported to {filename}")

        return filename
    except Exception as e:
        if log_event_callback:
            log_event_callback(f"Failed to export statistics: {str(e)}")
        return None