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
from predict import load_ensemble_model, load_single_model, preprocess_image, predict
from utils import get_device
from license_plate_ocr import extract_license_plates, initialize_ocr

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
initialize_ocr(['en'])

# Load model
device = get_device()
ENSEMBLE_MODEL_PATH = 'models/ensemble_model.pth'
IMPROVED_MODEL_PATH = 'models/improved_model_final.pth'
ensemble_model, class_to_idx = None, None
improved_model, improved_class_to_idx = None, None

def load_models():
    global ensemble_model, class_to_idx, improved_model, improved_class_to_idx
    try:
        # Load ensemble model
        ensemble_model, class_to_idx = load_ensemble_model(ENSEMBLE_MODEL_PATH, device)
        print("Ensemble model loaded successfully")
        print(f"Class to idx mapping: {class_to_idx}")
        
        # Load improved model
        improved_model, improved_class_to_idx = load_single_model(IMPROVED_MODEL_PATH, device)
        print("Improved model loaded successfully")
        print(f"Improved model class to idx mapping: {improved_class_to_idx}")
        
        # Ensure we have the correct mapping
        # The model expects: {'front': 0, 'left': 1, 'rear': 2, 'right': 3}
        expected_mapping = {'front': 0, 'left': 1, 'rear': 2, 'right': 3}
        
        # If the mapping is different, create a new mapping
        if class_to_idx != expected_mapping:
            print(f"Updating ensemble class mapping from {class_to_idx} to {expected_mapping}")
            class_to_idx = expected_mapping
            
        if improved_class_to_idx != expected_mapping:
            print(f"Updating improved model class mapping from {improved_class_to_idx} to {expected_mapping}")
            improved_class_to_idx = expected_mapping
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def verify_orientation(image_path, expected_orientation):
    """Verify if the image matches the expected orientation using dual model approach"""
    if ensemble_model is None or improved_model is None:
        print("Models not loaded")
        return False, 0.0, None, {}
    
    try:
        # Preprocess image
        image_tensor, _ = preprocess_image(image_path, image_size=320)
        
        # Make predictions with both models
        ensemble_pred_class, ensemble_confidence = predict(ensemble_model, image_tensor, class_to_idx, device)
        improved_pred_class, improved_confidence = predict(improved_model, image_tensor, improved_class_to_idx, device)
        
        print(f"Image: {image_path}, Expected: {expected_orientation}")
        print(f"Ensemble model prediction: {ensemble_pred_class}, Confidence: {ensemble_confidence}")
        print(f"Improved model prediction: {improved_pred_class}, Confidence: {improved_confidence}")
        
        # Create a comparison result
        model_comparison = {
            'ensemble': {
                'prediction': ensemble_pred_class,
                'confidence': ensemble_confidence
            },
            'improved': {
                'prediction': improved_pred_class,
                'confidence': improved_confidence
            }
        }
        
        # Determine final prediction
        # If both models agree, use that prediction
        if ensemble_pred_class == improved_pred_class:
            final_pred_class = ensemble_pred_class
            # Use the higher confidence score
            final_confidence = max(ensemble_confidence, improved_confidence)
        else:
            # If models disagree, use the one with higher confidence
            if ensemble_confidence > improved_confidence:
                final_pred_class = ensemble_pred_class
                final_confidence = ensemble_confidence
            else:
                final_pred_class = improved_pred_class
                final_confidence = improved_confidence
        
        # Check if prediction matches expected orientation
        # Convert both to lowercase for case-insensitive comparison
        is_match = final_pred_class.lower() == expected_orientation.lower()
        return is_match, final_confidence, final_pred_class, model_comparison
    except Exception as e:
        print(f"Verification error: {str(e)}")
        traceback.print_exc()
        return False, 0.0, None, {}

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
    detect_multiple = request.form.get('detect_multiple', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(f"temp_{int(time.time())}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', filename)
            file.save(filepath)
            
            print(f"Processing image: {filepath}, Expected orientation: {orientation}, Detect multiple: {detect_multiple}")
            
            # Verify orientation with dual model approach
            is_correct, confidence, predicted_class, model_comparison = verify_orientation(filepath, orientation)
            
            # Always detect multiple license plates for all views
            multiple_plates = extract_license_plates(filepath, detect_multiple=True)
            license_plate = multiple_plates[0] if multiple_plates else None
            
            # Return results with multiple plates
            return jsonify({
                'is_correct': is_correct,
                'confidence': confidence,
                'license_plate': license_plate,
                'multiple_plates': multiple_plates,
                'predicted_class': predicted_class,
                'model_comparison': model_comparison
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
                
                # Verify orientation with dual model approach
                is_correct, confidence, predicted_class, model_comparison = verify_orientation(filepath, orientations[orientation])
                
                verification_results[orientation] = {
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'filepath': filepath,
                    'predicted_class': predicted_class,
                    'model_comparison': model_comparison
                }
                
                # Extract license plate - prioritize front and rear views for license plate detection
                if orientation in ['front', 'rear']:
                    multiple_plates = extract_license_plates(filepath, detect_multiple=True)
                    if multiple_plates:
                        print(f"Detected license plates from {orientation} view: {multiple_plates}")
                        license_plates[orientation] = multiple_plates[0]
                else:
                    # For left/right views, still attempt detection but with multiple plates
                    multiple_plates = extract_license_plates(filepath, detect_multiple=True)
                    if multiple_plates:
                        print(f"Detected license plates from {orientation} view: {multiple_plates}")
                        license_plates[orientation] = multiple_plates[0]
                
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
                if image_type == 'towing_rear':
                    # For towing truck image, try to detect multiple plates
                    multiple_plates = extract_license_plates(filepath, detect_multiple=True)
                    
                    if multiple_plates and len(multiple_plates) >= 2:
                        # If we found at least 2 plates, the top one is the towed vehicle, bottom is the towing truck
                        license_plates['towed_vehicle_from_rear'] = multiple_plates[0]
                        license_plates['towing_truck'] = multiple_plates[1]
                        # Also store the single plate for backward compatibility
                        license_plates[image_type] = multiple_plates[1]  # Use the truck plate as the primary one
                    elif multiple_plates and len(multiple_plates) == 1:
                        # If we only found one plate, assume it's the towing truck
                        license_plates['towing_truck'] = multiple_plates[0]
                        license_plates[image_type] = multiple_plates[0]
                    else:
                        license_plates[image_type] = None
                else:
                    # For towed vehicle image, just detect a single plate
                    license_plate = extract_license_plates(filepath)
                    license_plates[image_type] = license_plate
                    if license_plate:
                        license_plates['towed_vehicle'] = license_plate
                
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
    
    # Get license plates
    vehicle_plates = session.get('license_plates', {})
    towing_plates = session.get('towing_license_plates', {})
    
    # Get front license plate from vehicle verification
    front_plate = vehicle_plates.get('front')
    
    # Get rear license plate from vehicle verification
    rear_plate = vehicle_plates.get('rear')
    
    # Get towed vehicle plate - prioritize the one from towing verification
    towed_vehicle_plate = towing_plates.get('towed_vehicle')
    
    # If no towed vehicle plate from towing verification, try the one detected from the rear image (top plate)
    if not towed_vehicle_plate:
        towed_vehicle_plate = towing_plates.get('towed_vehicle_from_rear')
    
    # If still no plate, try the one from vehicle verification (front or rear)
    if not towed_vehicle_plate:
        # Find most common license plate from vehicle images
        all_plates = [plate for plate in vehicle_plates.values() if plate]
        towed_vehicle_plate = max(set(all_plates), key=all_plates.count) if all_plates else None
    
    # Get towing truck plate
    towing_truck_plate = towing_plates.get('towing_truck')
    
    # If no specific towing truck plate, fall back to the towing_rear plate
    if not towing_truck_plate:
        towing_truck_plate = towing_plates.get('towing_rear')
    
    # Print debug info
    print(f"Vehicle plates: {vehicle_plates}")
    print(f"Towing plates: {towing_plates}")
    print(f"Front plate: {front_plate}")
    print(f"Rear plate: {rear_plate}")
    print(f"Towed vehicle plate: {towed_vehicle_plate}")
    print(f"Towing truck plate: {towing_truck_plate}")
    
    return render_template('complete.html',
                          vehicle_verified=vehicle_verified,
                          front_plate=front_plate,
                          rear_plate=rear_plate,
                          vehicle_plate=towed_vehicle_plate,
                          towing_truck_plate=towing_truck_plate)

if __name__ == '__main__':
    # Load models before starting the app
    if load_models():
        app.run(host='0.0.0.0', port=5002, debug=True)
    else:
        print("Failed to load models. Exiting.") 