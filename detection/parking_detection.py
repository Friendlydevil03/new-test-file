import cv2
import numpy as np
import cvzone


def process_parking_frame(frame, gpu_adaptive_threshold, cv_gpu_available=False):
    """Process frame for parking detection"""
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    # Use GPU accelerated threshold if available
    imgThreshold = gpu_adaptive_threshold(
        imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16, cv_gpu_available)

    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    return imgDilate


def check_parking_space(processed_img, img, parking_positions, threshold):
    """Check parking spaces in the processed image"""
    space_counter = 0
    for i, (x, y, w, h) in enumerate(parking_positions):
        # Ensure coordinates are within image bounds
        if (y >= 0 and y + h < processed_img.shape[0] and
                x >= 0 and x + w < processed_img.shape[1]):

            # Draw ID number for each space
            cv2.putText(img, str(i), (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            imgCrop = processed_img[y:y + h, x:x + w]
            count = cv2.countNonZero(imgCrop)

            if count < threshold:
                color = (0, 255, 0)  # Green for free
                space_counter += 1
            else:
                color = (0, 0, 255)  # Red for occupied

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cvzone.putTextRect(img, str(count), (x, y + h - 3), scale=1, thickness=2, offset=0)

    return img, space_counter