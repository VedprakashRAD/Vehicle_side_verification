import cv2
import numpy as np
import easyocr
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('license_plate_ocr')

# Initialize OCR reader once as a global variable
reader = None

def initialize_ocr(languages=['en']):
    """Initialize the OCR reader with specified languages"""
    global reader
    if reader is None:
        logger.info(f"Initializing EasyOCR with languages: {languages}")
        reader = easyocr.Reader(languages)
    return reader

def preprocess_image(image):
    """
    Preprocess the image to enhance license plate detection
    
    Args:
        image: OpenCV image object
        
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Apply adaptive threshold to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Create a list of preprocessed images to try
    preprocessed_images = [
        image,  # Original image
        gray,   # Grayscale
        bilateral,  # Bilateral filtered
        thresh  # Thresholded
    ]
    
    return preprocessed_images

def detect_license_plate_regions(image):
    """
    Attempt to detect regions that might contain license plates
    
    Args:
        image: OpenCV image object
        
    Returns:
        List of potential license plate regions (cropped images)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edges = cv2.Canny(bilateral, 30, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    potential_regions = []
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # License plates are typically rectangles with 4 corners
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio typical for license plates (2:1 to 5:1)
            aspect_ratio = float(w) / h
            if 1.5 <= aspect_ratio <= 5.0:
                # Extract the region
                plate_region = image[y:y+h, x:x+w]
                potential_regions.append(plate_region)
    
    # If no regions found with contour method, return the whole image
    if not potential_regions:
        potential_regions.append(image)
        
    return potential_regions

def is_valid_license_plate(text):
    """
    Check if the detected text is likely a valid license plate
    
    Args:
        text: Detected text string
        
    Returns:
        Boolean indicating if text is likely a license plate
    """
    # Remove spaces and special characters
    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Check if length is appropriate for a license plate (4-10 characters)
    if not (4 <= len(cleaned_text) <= 10):
        return False
    
    # Check if it has both letters and numbers (common for license plates)
    has_letters = bool(re.search(r'[A-Z]', cleaned_text))
    has_numbers = bool(re.search(r'[0-9]', cleaned_text))
    
    # Most license plates have both letters and numbers, but we'll be more lenient
    # and accept plates with just letters or just numbers if they're long enough
    if not (has_letters or has_numbers):
        return False
    
    return True

def extract_license_plates(image_path, detect_multiple=False):
    """
    Extract license plate text from an image with improved accuracy
    
    Args:
        image_path: Path to the image
        detect_multiple: If True, attempt to detect multiple plates
        
    Returns:
        For single plate detection: The detected license plate text or None
        For multiple plate detection: A list of plates ordered by vertical position (top to bottom)
    """
    try:
        # Initialize OCR if not already done
        initialize_ocr()
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return None if not detect_multiple else []
        
        # Store original image dimensions for logging
        original_height, original_width = image.shape[:2]
        logger.info(f"Processing image: {image_path} ({original_width}x{original_height})")
        
        # Preprocess image to get multiple versions to try
        preprocessed_images = preprocess_image(image)
        
        # Try to detect license plate regions
        potential_regions = detect_license_plate_regions(image)
        
        # Add the potential regions to our list of images to process
        all_images_to_process = preprocessed_images + potential_regions
        
        # Store all detected license plates
        all_detections = []
        
        # Process each image version
        for idx, img in enumerate(all_images_to_process):
            try:
                # Perform OCR
                results = reader.readtext(img)
                
                # Process OCR results
                for (bbox, text, prob) in results:
                    # Clean the text
                    cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
                    
                    # Skip if too short or confidence too low
                    if len(cleaned_text) < 4 or prob < 0.3:  # Lower the threshold to catch more plates
                        continue
                    
                    # Get position information
                    if len(bbox) >= 2:  # Ensure bbox has at least 2 points
                        top_left, _, _, bottom_right = bbox
                        y_position = (top_left[1] + bottom_right[1]) / 2
                    else:
                        # If bbox format is unexpected, use a default position
                        y_position = 0
                    
                    # Add to detections if it looks like a license plate
                    if is_valid_license_plate(cleaned_text):
                        all_detections.append((cleaned_text, prob, y_position))
                        logger.info(f"Detected potential plate: {cleaned_text} (confidence: {prob:.2f})")
            except Exception as e:
                logger.warning(f"Error processing image variant {idx}: {str(e)}")
                continue
        
        # If no valid detections, return None/empty list
        if not all_detections:
            logger.info("No valid license plates detected")
            return None if not detect_multiple else []
        
        # Remove duplicates (keeping highest confidence)
        unique_detections = {}
        for text, prob, y_pos in all_detections:
            if text not in unique_detections or prob > unique_detections[text][0]:
                unique_detections[text] = (prob, y_pos)
        
        # Convert back to list format
        filtered_detections = [(text, prob, y_pos) for text, (prob, y_pos) in unique_detections.items()]
        
        # For single plate detection, return the highest confidence plate
        if not detect_multiple:
            filtered_detections.sort(key=lambda x: x[1], reverse=True)
            best_detection = filtered_detections[0][0]
            logger.info(f"Selected best license plate: {best_detection}")
            return best_detection
        
        # For multiple plate detection, sort by vertical position (top to bottom)
        filtered_detections.sort(key=lambda x: x[2])
        result = [detection[0] for detection in filtered_detections]
        
        logger.info(f"Returning {len(result)} license plates in top-to-bottom order")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting license plate: {str(e)}", exc_info=True)
        return None if not detect_multiple else []

# For testing the module independently
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python license_plate_ocr.py <image_path> [detect_multiple]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_multiple = len(sys.argv) > 2 and sys.argv[2].lower() == 'true'
    
    result = extract_license_plates(image_path, detect_multiple)
    
    if detect_multiple:
        print(f"Detected {len(result)} license plates:")
        for i, plate in enumerate(result):
            print(f"{i+1}. {plate}")
    else:
        print(f"Detected license plate: {result}") 