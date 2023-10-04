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
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
ocr_cache = {}
# Initialize the Flask app and Firebase
app = Flask(__name__, template_folder='templates')
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

@app.route('/adminlogin')
def admin_login():
    return render_template('adminlogin.html')

@app.route('/userlogin')
def user_login():
    return render_template('userlogin.html')

@app.route('/register')  # This route is for the "Register new User" page
def user_register():
    return render_template('register.html')

@app.route('/adminhome')  # This route is for the "Register new User" page
def admin_home():
    return render_template('adminhome.html')

@app.route('/userhome')  # This route is for the "Register new User" page
def user_home():
    return render_template('userhome.html')

@app.route('/view-files')  # This route is for the "Register new User" page
def allfiles():
    return render_template('viewallfiles.html')

@app.route('/view-all-files', methods=['GET'])
def view_all_files():
    # List the files in the Firebase Storage bucket
    blobs = bucket.list_blobs()

    # Create a list to store file information
    file_info_list = []

    for blob in blobs:
        # Generate a signed URL for the file
        signed_url = blob.generate_signed_url(
            version='v4',
            expiration=timedelta(hours=1),
            method='GET'
        )
        
        # Append file information to the list
        file_info_list.append({
            'file_name': blob.name,
            'signed_url': signed_url
        })

    print("File Info List:", file_info_list)  # Add this line for debugging
    
    return jsonify(file_info_list)

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

def perform_ocr_on_image(image, languages):
    try:
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'  # Adjust the path if needed
        enhanced_image = enhance_image_ocr(image)
        text = pytesseract.image_to_string(enhanced_image, lang=languages, config='--psm 6')
    except Exception as e:
        print("Error performing OCR:", e)
        text = ""
    return text



# Function to perform enhanced OCR on a single file
def perform_ocr_for_file(blob, search_query, languages):
    file_name = blob.name

    # Check if the OCR result is already in the cache
    if file_name in ocr_cache:
        text = ocr_cache[file_name]
        signed_url = None  # Set signed_url to None if the result is in the cache
    else:
        # If not in cache, perform OCR
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
                text = perform_ocr_on_image(image, languages)
            except Exception as e:
                print("Error processing image:", e)
                text = ""
        else:
            text = ""

        text = preprocess_text(text)

        # Store the result in the cache
        ocr_cache[file_name] = text.lower()

    return text.lower(), {
        'file_name': file_name,
        'signed_url': signed_url
    }
def clear_ocr_cache():
    global ocr_cache
    ocr_cache = {}

# Add a route to clear the cache
@app.route('/clear-cache', methods=['GET'])
def clear_cache():
    clear_ocr_cache()
    return "Cache cleared."

@app.route('/perform-ocr', methods=['POST'])
def perform_ocr():
    search_query = request.json['searchQuery']
    languages = 'eng+hin+mar+guj'
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
            futures.append(executor.submit(perform_ocr_for_file, blob, search_query, languages))

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