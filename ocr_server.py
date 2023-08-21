# Import the required libraries
from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, storage
from PyPDF2 import PdfReader
import io
from datetime import timedelta
import requests
from docx import Document
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor for parallel processing
from sentence_transformers import SentenceTransformer, util
import re
from fuzzywuzzy import fuzz
import pytessFueract
from PIL import Image, ImageEnhance, ImageFilter


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Initialize the Flask app and Firebase
app = Flask(__name__)
cred = credentials.Certificate('./ocrscanner-887f5-firebase-adminsdk-rejm0-d4d6480be1.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'ocrscanner-887f5.appspot.com'})
bucket = storage.bucket()

# Fuzzy matching threshold
FUZZY_THRESHOLD = 70  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ocr')
def ocr():
    return render_template('ocr.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

def preprocess_text(text):
    # Remove extra whitespaces and newlines
    text = ' '.join(text.split())
    return text

# Function to enhance OCR on an image
def enhance_image_ocr(image):
    # Enhance image contrast and sharpness
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2.0)  # Adjust the factor as needed
    enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
    return enhanced_image

def perform_ocr_on_image(image):
    try:
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path if needed
        enhanced_image = enhance_image_ocr(image)
        text = pytesseract.image_to_string(enhanced_image, lang='eng', config='--psm 6')
    except Exception as e:
        print("Error performing OCR:", e)
        text = ""
    return text


# Function to perform enhanced OCR on a single file
def perform_ocr_for_file(blob, search_query):
    file_name = blob.name
    signed_url = blob.generate_signed_url(
        version='v4',
        expiration=timedelta(hours=1),
        method='GET'
    )

    response = requests.get(signed_url)
    file_data = response.content

    if blob.content_type == 'application/pdf':
        pdf = PdfReader(io.BytesIO(file_data))
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    elif blob.content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = Document(io.BytesIO(file_data))
        text = ' '.join([para.text for para in doc.paragraphs])
    elif blob.content_type.startswith('image/'):
        try:
            image = Image.open(io.BytesIO(file_data))
            text = perform_ocr_on_image(image)
        except Exception as e:
            print("Error processing image:", e)
            text = ""
    else:
        text = ""

    text = preprocess_text(text)
    return text.lower(), {
        'file_name': file_name,
        'signed_url': signed_url
    }

@app.route('/perform-ocr', methods=['POST'])
def perform_ocr():
    search_query = request.json['searchQuery']
    categorized_links = {
        'PDF': [],
        'WORD': [],
        'IMAGES': []
    }

    # List the files in the Firebase Storage bucket
    blobs = bucket.list_blobs()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for blob in blobs:
            futures.append(executor.submit(perform_ocr_for_file, blob, search_query))

        for future in futures:
            text, link_data = future.result()

            if text and (search_query.lower() in text or fuzz.partial_ratio(search_query.lower(), text) >= FUZZY_THRESHOLD):
                if link_data['file_name'].endswith('.pdf'):
                    categorized_links['PDF'].append(link_data)
                elif link_data['file_name'].endswith('.docx'):
                    categorized_links['WORD'].append(link_data)
                elif link_data['file_name'].endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    categorized_links['IMAGES'].append(link_data)

    return jsonify(categorized_links)

if __name__ == '__main__':
    app.run(debug=True)