import os
import torch
import numpy as np
from PIL import Image
import cv2
import easyocr
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
import time
from datetime import datetime
import re
import traceback

# Import from existing project files
from predict import load_ensemble_model, preprocess_image, predict
from utils import get_device

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'vehicle_verification_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'vehicle'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'towing'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'temp'), exist_ok=True)  # For temporary files

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load model
device = get_device()
MODEL_PATH = 'models/ensemble_model.pth'
model, class_to_idx = None, None

def load_model():
    global model, class_to_idx
    try:
        model, class_to_idx = load_ensemble_model(MODEL_PATH, device)
        print("Model loaded successfully")
        print(f"Class to idx mapping: {class_to_idx}")
        
        # Ensure we have the correct mapping
        # The model expects: {'front': 0, 'left': 1, 'rear': 2, 'right': 3}
        expected_mapping = {'front': 0, 'left': 1, 'rear': 2, 'right': 3}
        
        # If the mapping is different, create a new mapping
        if class_to_idx != expected_mapping:
            print(f"Updating class mapping from {class_to_idx} to {expected_mapping}")
            class_to_idx = expected_mapping
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_license_plate(image_path):
    """Extract license plate text from an image using EasyOCR"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Perform OCR
        results = reader.readtext(image)
        
        # Process OCR results
        license_plates = []
        for (bbox, text, prob) in results:
            # Filter for potential license plates (alphanumeric, length between 5-10)
            cleaned_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(cleaned_text) >= 5 and len(cleaned_text) <= 10 and prob > 0.5:
                license_plates.append((cleaned_text, prob))
        
        # Return the highest confidence result
        if license_plates:
            license_plates.sort(key=lambda x: x[1], reverse=True)
            return license_plates[0][0]
        
        return None
    except Exception as e:
        print(f"OCR error: {e}")
        return None

def verify_orientation(image_path, expected_orientation):
    """Verify if the image matches the expected orientation"""
    if model is None:
        print("Model not loaded")
        return False, 0.0, None
    
    try:
        # Preprocess image
        image_tensor, _ = preprocess_image(image_path, image_size=320)
        
        # Make prediction
        pred_class, confidence = predict(model, image_tensor, class_to_idx, device)
        
        print(f"Image: {image_path}, Expected: {expected_orientation}, Predicted: {pred_class}, Confidence: {confidence}")
        
        # Check if prediction matches expected orientation
        # Convert both to lowercase for case-insensitive comparison
        is_match = pred_class.lower() == expected_orientation.lower()
        return is_match, confidence, pred_class
    except Exception as e:
        print(f"Verification error: {str(e)}")
        traceback.print_exc()
        return False, 0.0, None

@app.route('/')
def index():
    # Clear any existing session data
    session.clear()
    return render_template('index.html')

@app.route('/verify-image', methods=['POST'])
def verify_image():
    """API endpoint for real-time image verification and license plate detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    orientation = request.form.get('orientation', '')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(f"temp_{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', filename)
            file.save(filepath)
            
            print(f"Processing image: {filepath}, Expected orientation: {orientation}")
            
            # Verify orientation
            is_correct, confidence, predicted_class = verify_orientation(filepath, orientation)
            
            # Extract license plate
            license_plate = extract_license_plate(filepath)
            
            # Return results
            return jsonify({
                'is_correct': is_correct,
                'confidence': confidence,
                'license_plate': license_plate,
                'predicted_class': predicted_class
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/vehicle-verification', methods=['GET', 'POST'])
def vehicle_verification():
    if request.method == 'POST':
        vehicle_images = {}
        license_plates = {}
        verification_results = {}
        
        # Expected orientations for each upload field
        orientations = {
            'front': 'front',
            'rear': 'rear',
            'left': 'left',
            'right': 'right'
        }
        
        # Process each orientation
        for orientation in orientations:
            # Check for manual override
            manual_override = request.form.get(f'{orientation}_verified') == 'true'
            
            if orientation not in request.files:
                flash(f'No {orientation} image uploaded')
                continue
                
            file = request.files[orientation]
            if file.filename == '':
                flash(f'No {orientation} image selected')
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{orientation}_{int(time.time())}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'vehicle', filename)
                file.save(filepath)
                
                # If manual override, consider it verified
                if manual_override:
                    is_correct = True
                    confidence = 1.0
                    predicted_class = orientation
                    print(f"Manual override for {orientation} image")
                else:
                    # Verify orientation
                    is_correct, confidence, predicted_class = verify_orientation(filepath, orientations[orientation])
                
                verification_results[orientation] = {
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'filepath': filepath,
                    'predicted_class': predicted_class,
                    'manual_override': manual_override
                }
                
                # Extract license plate
                license_plate = extract_license_plate(filepath)
                license_plates[orientation] = license_plate
                
                vehicle_images[orientation] = filename
        
        # Store results in session
        session['vehicle_images'] = vehicle_images
        session['license_plates'] = license_plates
        session['verification_results'] = verification_results
        
        return redirect(url_for('vehicle_results'))
    
    return render_template('vehicle_verification.html')

@app.route('/vehicle-results')
def vehicle_results():
    if 'vehicle_images' not in session:
        return redirect(url_for('vehicle_verification'))
    
    return render_template('vehicle_results.html', 
                          images=session['vehicle_images'],
                          license_plates=session['license_plates'],
                          verification=session['verification_results'])

@app.route('/towing-verification', methods=['GET', 'POST'])
def towing_verification():
    if request.method == 'POST':
        towing_images = {}
        license_plates = {}
        
        # Process towing images
        for image_type in ['towing_front', 'towing_rear']:
            if image_type not in request.files:
                flash(f'No {image_type} image uploaded')
                continue
                
            file = request.files[image_type]
            if file.filename == '':
                flash(f'No {image_type} image selected')
                continue
                
            if file and allowed_file(file.filename):
                filename = secure_filename(f"{image_type}_{int(time.time())}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'towing', filename)
                file.save(filepath)
                
                # Extract license plate
                license_plate = extract_license_plate(filepath)
                license_plates[image_type] = license_plate
                
                towing_images[image_type] = filename
        
        # Store results in session
        session['towing_images'] = towing_images
        session['towing_license_plates'] = license_plates
        
        return redirect(url_for('towing_results'))
    
    return render_template('towing_verification.html')

@app.route('/towing-results')
def towing_results():
    if 'towing_images' not in session:
        return redirect(url_for('towing_verification'))
    
    return render_template('towing_results.html', 
                          images=session['towing_images'],
                          license_plates=session['towing_license_plates'])

@app.route('/complete')
def complete():
    # Combine all verification results
    vehicle_verified = False
    if 'verification_results' in session:
        # Check if all orientations were correctly verified
        results = session.get('verification_results', {})
        if results and all(r.get('is_correct', False) for r in results.values()):
            vehicle_verified = True
    
    # Get consistent license plate
    vehicle_plates = session.get('license_plates', {})
    towing_plates = session.get('towing_license_plates', {})
    
    # Find most common license plate from vehicle images
    all_plates = [plate for plate in vehicle_plates.values() if plate]
    vehicle_plate = max(set(all_plates), key=all_plates.count) if all_plates else None
    
    # Get towing truck plate directly from the towing_rear image
    towing_truck_plate = towing_plates.get('towing_rear')
    
    # If no towing_rear plate, check towing_front as a fallback
    if not towing_truck_plate:
        towing_truck_plate = towing_plates.get('towing_front')
    
    # Print debug info
    print(f"Vehicle plates: {vehicle_plates}")
    print(f"Towing plates: {towing_plates}")
    print(f"Selected vehicle plate: {vehicle_plate}")
    print(f"Selected towing truck plate: {towing_truck_plate}")
    
    return render_template('complete.html',
                          vehicle_verified=vehicle_verified,
                          vehicle_plate=vehicle_plate,
                          towing_truck_plate=towing_truck_plate)

if __name__ == '__main__':
    # Load model before starting the app
    if load_model():
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Failed to load model. Exiting.") 