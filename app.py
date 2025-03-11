from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image  # For resizing

app = Flask(__name__)
# app.secret_key = "your_secret_key"  # Uncomment if needed for sessions
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('model/insect_model.keras')

class_names = [
    'Africanized Honey Bees (Killer Bees)', 'Aphids', 'Armyworms', 
    'Brown Marmorated Stink Bugs', 'Cabbage Loopers', 'Citrus Canker', 
    'Colorado Potato Beetles', 'Corn Borers', 'Corn Earworms', 
    'Fall Armyworms', 'Fruit Flies', 'Spider Mites', 'Thrips', 
    'Tomato Hornworms', 'Western Corn Rootworms'
]

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/classify')
def classify():
    return render_template('classify.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with Image.open(file_path) as img:
            img = img.resize((224, 224))  # Resize the image
            img.save(file_path)

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict([img_array, img_array])
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_label = class_names[predicted_class]

        return render_template('result.html', prediction=predicted_label, image_file=filename)

    return redirect(url_for('classify'))

if __name__ == '__main__':
    app.run(debug=True)
