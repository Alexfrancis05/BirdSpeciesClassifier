# Birds Classification Web App using Flask and TensorFlow
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Define upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
MODEL_PATH = 'D:/Birds_classification/model/bird_species_classifier.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")
model = load_model(MODEL_PATH)

# Image size used during training
IMAGE_SIZE = (224, 224)

# Define bird species list
BIRD_SPECIES = [
    'Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove-billed Ani',
    'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird',
    'Red-winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink',
    'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird',
    'Gray Catbird', 'Yellow-breasted Chat', 'Eastern Towhee', 'Chuck-will’s-widow',
    'Brandt Cormorant', 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird',
    'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black-billed Cuckoo',
    'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned Rosy-Finch',
    'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher',
    'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher',
    'Vermilion Flycatcher', 'Yellow-bellied Flycatcher', 'Frigatebird',
    'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch',
    'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe',
    'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak',
    'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull',
    'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull',
    'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird',
    'Rufous Hummingbird', 'Green Violetear', 'Long-tailed Jaeger', 'Pomarine Jaeger',
    'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird',
    'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher',
    'Ringed Kingfisher', 'White-breasted Kingfisher', 'Red-legged Kittiwake',
    'Horned Lark', 'Pacific Loon', 'Common Loon', 'Mallard', 'Western Meadowlark',
    'Hooded Merganser', 'Red-breasted Merganser', 'Mockingbird', 'Nighthawk',
    'Clark Nutcracker', 'White-breasted Nuthatch', 'Baltimore Oriole',
    'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole'
]


# Create class indices dictionary
class_indices = {i: species for i, species in enumerate(BIRD_SPECIES)}

# Simple in-memory user storage
users = {}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess and predict the image
def predict_image(img):
    img = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return class_indices[predicted_class], confidence

# Home route
@app.route('/')
def home():
    if 'logged_in' in session:
        return redirect(url_for('classify'))
    return render_template('index.html')

# About route
@app.route('/about')
def about():
    return render_template('about.html')

# Projects route
@app.route('/projects')
def projects():
    return render_template('projects.html')

# Blog route
@app.route('/blog')
def blog():
    return render_template('blog.html')

# Contact route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and users[username]['password'] == password:
            session['logged_in'] = True
            return redirect(url_for('classify'))
        else:
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if username in users:
            return render_template('register.html', error='Username already exists')
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        users[username] = {'email': email, 'password': password}
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Classify route
@app.route('/classify')
def classify():
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    return render_template('classify.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:  # Handle file upload
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            img = Image.open(filepath)
            predicted_species, confidence = predict_image(img)
            
            return jsonify({
                'species': predicted_species,
                'confidence': confidence,
                'image_path': filepath
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    elif request.json and 'image' in request.json:  # Handle webcam frame
        try:
            img_data = request.json['image']
            img_data = img_data.split(',')[1]  # Remove "data:image/jpeg;base64,"
            img_bytes = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_bytes)).convert('RGB')
            
            predicted_species, confidence = predict_image(img)
            
            return jsonify({
                'species': predicted_species,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
    
    return jsonify({'error': 'No file or image data provided'}), 400

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)