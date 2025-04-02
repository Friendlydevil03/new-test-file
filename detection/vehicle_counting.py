import cv2
import numpy as np
import cvzone

def get_centroid(x, y, w, h):
    """Calculate centroid of a rectangle"""
    return x + w // 2, y + h // 2

def detect_vehicles_traditional(frame1, frame2, min_contour_width, min_contour_height, line_height, offset, matches, vehicle_counter):
    """Process frames to detect and count vehicles using traditional computer vision"""
    # Get difference between frames
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(grey, (5, 5), 0)

    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw detection line
    line_y = line_height
    if line_y >= frame1.shape[0]:
        line_y = frame1.shape[0] - 50
    cv2.line(frame1, (0, line_y), (frame1.shape[1], line_y), (0, 255, 0), 2)

    # Process contours
    new_matches = matches.copy()
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)

        if not contour_valid:
            continue

        cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)

        centroid = get_centroid(x, y, w, h)
        new_matches.append(centroid)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

    # Check for vehicles crossing the line
    new_counter = vehicle_counter
    remaining_matches = []
    for (x, y) in new_matches:
        if (line_y - offset) < y < (line_y + offset):
            new_counter += 1
        else:
            remaining_matches.append((x, y))

    # Display count
    cvzone.putTextRect(frame1, f"Vehicle Count: {new_counter}", (10, 30),
                       scale=2, thickness=2, offset=10, colorR=(0, 200, 0))

    return frame1, remaining_matches, new_counter

def detect_vehicles_ml(frame, ml_detector, matches, vehicle_counter, line_height, offset):
    """Process frame with ML detection"""
    try:
        # Get vehicle detections
        detections = ml_detector.detect_vehicles(frame)

        # Draw detection line
        line_y = line_height
        if line_y >= frame.shape[0]:
            line_y = frame.shape[0] - 50
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

        # Process detections
        current_centroids = []
        for x1, y1, x2, y2, score, label in detections:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate centroid
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            centroid = (centroid_x, centroid_y)

            # Add to current centroids
            current_centroids.append(centroid)

            # Draw centroid
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

            # Label with confidence
            label_text = f"{ml_detector.classes[label]}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count vehicles crossing the line
        new_counter = vehicle_counter
        for centroid in current_centroids:
            if (line_y - offset) < centroid[1] < (line_y + offset):
                if centroid not in matches:
                    new_counter += 1
                    matches.append(centroid)

        # Clean up old centroids
        new_matches = [match for match in matches
                      if any(np.linalg.norm(np.array(match) - np.array(c)) < 50 for c in current_centroids)]

        # Display count
        cvzone.putTextRect(frame, f"Vehicle Count: {new_counter}", (10, 30),
                           scale=2, thickness=2, offset=10, colorR=(0, 200, 0))

        return frame, new_matches, new_counter
    except Exception as e:
        print(f"ML detection error: {str(e)}")
        return frame, matches, vehicle_counter