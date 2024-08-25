from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory

app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')

# Class labels
class_labels = {
    0: 'Bacterial_Infections',
    1: 'Gall_Disease', 
    2: 'Healthy',
    3: 'Peach_Leaf_Curl',
    4: 'Powdery',
    5: 'Rust',
    6: 'Wilting'
}

# Dictionary of solutions for each disease
disease_solutions = {
    'Wilting': [    
        'Why did this happen? :- Occurs due to various factors such as fungal diseases, bacterial infections, water stress, or environmental factors, leading to loss of turgor pressure in plants and subsequent drooping or wilting of leaves.',
        'Or a Environmental_Stress',
        'Sub parts :- Fusarium wilt, Bacterial wilt, Verticillium wilt, Drought Stress, Heat Stress, Cold Stress, Salinity Stress, Pollution Stress',
        'Ensure proper and consistent watering based on plant needs.',
        'Improve soil drainage to prevent waterlogging.',
        'Identify and address underlying diseases promptly.',
        'Protect plants from extreme temperatures and environmental stressors.'
    ],
    'Rust': [
        'Why did this happen? :- Caused by fungal pathogens belonging to the order Pucciniales, resulting in rust-colored lesions on plant leaves, stems, or fruits.',
        'Sub parts :- Plant rust diseases(e.g. wheat rust, coffee rust) ,Iron rust ',
        'Apply fungicides labeled for rust control as directed.',
        'Prune and dispose of infected plant parts during the dormant season.',
        'Opt for plant varieties that show resistance or tolerance to rust.',
        'Maintain good plant spacing and air circulation.'
    ],

    'Powdery': [
        'Why did this happen? :-  Refers to powdery mildew, a fungal disease affecting plants, characterized by the presence of white powdery patches on leaves and stems.',
        'Sub parts :- Powdery mildew ,Talcum powder exposure ',
        'Apply fungicides designed to combat powdery mildew.',
        'Remove and dispose of infected plant parts to reduce spread.',
        'Choose plant varieties resistant to powdery mildew.',
        'Optimize plant spacing, avoid overhead irrigation, and provide adequate sunlight.'
    ],
    'Peach_Leaf_Curl': [
        'Why did this happen? :- Caused by fungal infection (Taphrina deformans) on peach trees, resulting in characteristic leaf curling and distortion.',
        'Sub Parts :- Fungal infections on peach trees, Leaf spot diseases, Bacterial canker',
        'Apply fungicides during the tree\'s dormant season.',
        'Remove and destroy infected leaves to reduce fungal inoculum.',
        'Minimize leaf wetness by using drip irrigation or applying water at the base.',
        'Choose peach or nectarine varieties resistant to peach leaf curl.'
    ],
    'Healthy': ['No action required. Your plant is healthy!',
                'Represents a state of being free from disease or illness, indicating normal physiological function and well-being.',
                ],
    
    'Gall_Disease': [
        'Why did this happen? :- Resulting from abnormal growths (galls) on plants induced by various pathogens, leading to tissue distortion and damage.',
        'Sub parts :- Rose_Gall, Gallstones, Cholecystitis, Choledocholithiasis, Gallbladder cancer',
        'Avoid injuries during cultivation to prevent infection.',
        'Use disease-free planting material and promote plant health.',
        'Promptly remove infected plants to prevent bacterial spread.',
        'Consider soil sterilization and antibacterial agents in affected areas.'
    ],

    'Bacterial_Infections': [
        'Why did this happen? :- Caused by the invasion and proliferation of harmful bacteria within the body, leading to tissue damage and disease.',
        'sub parts :- Tuberculosis ,Pneumonia ,Urinary tract infections (UTIs), Staph infections ,E. coli infection',
        'Practice preventive measures like crop rotation and proper spacing.',
        'Use targeted treatments with copper-based sprays or natural antibacterial solutions.',
        'Timely pruning of infected parts is essential.',
        'Maintain good plant hygiene for effective bacterial infection management.'
    ]
}

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x


def is_leaf(image_path):
    # Use OpenCV to read the image
    image = cv2.imread(image_path)
    if image is None:
        return False

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use contour analysis to check if the image has a leaf-like shape
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If there is at least one contour, consider it a leaf
    return len(contours) > 0

def detect_disease(image_path):
    # Check if the file extension is allowed
    allowed_extensions = {'jpg', 'jpeg', 'png','webp'}
    if not image_path.lower().endswith(tuple(allowed_extensions)):
        return "Invalid file type. Please select a valid image file (JPEG or PNG).", None, None

    # Check if the image is a leaf
    if not is_leaf(image_path):
        return "The selected image does not appear to be a leaf photo.", None, None

    # Preprocess the image
    x = preprocess_image(image_path)

    # Make predictions
    predictions = model.predict(x)

    # Get the predicted class index
    predicted_class = np.argmax(predictions)

    # Map the class index to the class label
    predicted_label = class_labels.get(predicted_class, 'Unknown')

    return None, predicted_label, x


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    image = request.files['image']
    image_path = 'static/uploads/' + image.filename
    image.save(image_path)

    error_message, predicted_disease, image_data = detect_disease(image_path)

    if error_message:
        return render_template('result.html', error_message=error_message)

    solution = disease_solutions.get(predicted_disease, 'No solution found.')

    return render_template('result.html', predicted_disease=predicted_disease, solution=solution, image_path=image_path)

@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory('videos', filename)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=True)