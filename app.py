from flask import Flask, request, render_template, jsonify
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import pdfplumber  # For extracting text from PDFs
import docx  # For extracting text from DOCX files
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

app = Flask(__name__)

# Load dataset
resumeDataSet = pd.read_csv("C:/Users/Arpita Patil/OneDrive/Pictures/Desktop/majorproject - final/templates/Cleaned_AugmentedResumeDataSet.csv")
resumeDataSet['cleaned_resume'] = ''

# Enhanced cleaning function for text queries
def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # Remove URLs
    resumeText = re.sub(r'RT|cc', ' ', resumeText)       # Remove RT and cc
    resumeText = re.sub(r'#\S+', '', resumeText)         # Remove hashtags
    resumeText = re.sub(r'@\S+', ' ', resumeText)        # Remove mentions
    resumeText = re.sub(r'[^\w\s]', ' ', resumeText)     # Remove special characters/punctuation
    resumeText = re.sub(r'\s+', ' ', resumeText)         # Remove extra whitespace
    resumeText = resumeText.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    return resumeText.strip()

# Apply cleaning function to the dataset
resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))

# TF-IDF Vectorization
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
word_vectorizer.fit(resumeDataSet['cleaned_resume'])
tfidf_matrix = word_vectorizer.transform(resumeDataSet['cleaned_resume'])

# Function to get job categories from the dataset
def get_job_categories():
    return sorted(resumeDataSet['Category'].dropna().unique().tolist())

# Function to validate if the file is a valid PDF
def is_valid_pdf(file):
    try:
        file.seek(0)  # Ensure we're reading from the start of the file
        first_bytes = file.read(4)
        return first_bytes == b'%PDF'
    except Exception as e:
        return False

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            if text:
                return text
            else:
                # If no text is found, attempt OCR fallback
                print("No text found in PDF, attempting OCR...")
                return extract_text_using_ocr(file)
    except Exception as e:
        return f"Error in PDF extraction: {str(e)}"  # Return error message

# Fallback method to use OCR for text extraction from image-based PDFs
def extract_text_using_ocr(file):
    try:
        # Convert the PDF pages to images
        print("Converting PDF to images for OCR...")
        pages = convert_from_path(file, 300)
        
        # Use OCR to extract text from the image
        text = ""
        for page in pages:
            print("Applying OCR to a page...")
            text += pytesseract.image_to_string(page)
        
        if text.strip() == "":
            print("OCR extracted no text. Possible issue with image quality.")
            return "OCR failed: No text extracted."
        return text
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        return f"OCR failed: {str(e)}"  # If OCR fails, return the error

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

# Enhanced function to extract details (Name, Phone, Email, Job Category, Resume Details)
def extract_resume_details(text):
    # Extract name (improved regex to capture variations)
    name = re.search(r'(?:Name[:\-]?\s*|\bFull\s*Name[:\-]?\s*)([A-Za-z\s]+)', text)
    
    # Extract phone number (improved to handle various formats)
    phone = re.search(r'(\+?\d{1,3}[-\s]?)?(\(?\d{3}\)?[-\s]?)?\d{3}[-\s]?\d{4}', text)
    
    # Extract email address (standard email regex)
    email = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    
    # Extract job category (search for typical terms like "Category", "Role", "Position", "Job Title")
    job_category = re.search(r'(?:Category[:\-]?\s*|Role[:\-]?\s*|Position[:\-]?\s*|Job\s*Title[:\-]?\s*)([A-Za-z\s]+)', text)
    
    # Extract the name, phone, email, and category if found, otherwise return "N/A"
    name = name.group(1) if name else 'N/A'
    phone = phone.group(0) if phone else 'N/A'
    email = email.group(0) if email else 'N/A'
    job_category = job_category.group(1) if job_category else 'N/A'

    # Extract resume details (we will return the first 100 words to get a better snippet)
    resume_details = ' '.join(text.split()[:100]) + '...'  # Preview first 100 words for better context

    # Return extracted details
    return {
        "Name": name,
        "Phone": phone,
        "Email": email,
        "Job Category": job_category,
        "Resume Details": resume_details
    }

# Find similar resumes with cleaned and truncated details for text/voice queries
def find_similar_resumes(query, tfidf_matrix, resume_data, top_n=5):
    query_vector = word_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    if similarities.max() == 0:
        return None  # No matches found

    top_indices = similarities.argsort()[-top_n:][::-1]
    results = []
    
    for i in top_indices:
        result = {
            "Name": str(resume_data.iloc[i]["Name"]) if "Name" in resume_data else "N/A",
            "Email": str(resume_data.iloc[i]["Email"]) if "Email" in resume_data else "N/A",
            "Phone": str(resume_data.iloc[i]["Phone"]) if "Phone" in resume_data else "N/A",
            "Category": str(resume_data.iloc[i]["Category"]) if "Category" in resume_data else "N/A",
            "Resume": ' '.join(cleanResume(resume_data.iloc[i]["Resume"]).split()[:50]) + '...',
            "Similarity": float(round(similarities[i], 2))
        }
        results.append(result)

    return results

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_resume():
    if "resume_file" not in request.files:
        return jsonify({"message": "No file uploaded."})

    file = request.files["resume_file"]
    if file.filename == "":
        return jsonify({"message": "No selected file."})

    try:
        print(f"Received file: {file.filename}")

        # Check if the uploaded file is a valid PDF
        if file.filename.endswith('.pdf'):
            if not is_valid_pdf(file):
                return jsonify({"message": "The uploaded file is not a valid PDF."})

            print("Processing PDF file...")
            text = extract_text_from_pdf(file)
        elif file.filename.endswith('.docx'):
            print("Processing DOCX file...")
            text = extract_text_from_docx(file)
        else:
            return jsonify({"message": "Unsupported file type. Please upload a PDF or DOCX file."})

        # Check if text extraction was successful or failed
        if "OCR failed" in text or "Error" in text:
            print(f"Text extraction failed: {text}")
            return jsonify({"message": f"Error processing file: {text}"})

        # Extract details like Name, Phone, Job Category, etc.
        print("Extracting resume details...")
        resume_details = extract_resume_details(text)

        print(f"Resume details extracted: {resume_details}")

        return jsonify({
            "resume_details": resume_details,
        })

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({"message": f"Error processing file: {str(e)}"})

@app.route("/query", methods=["POST"])
def handle_query():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"message": "Query is empty. Please provide a valid search query."})

    # Find the top 5 similar resumes based on the query
    similar_resumes = find_similar_resumes(query, tfidf_matrix, resumeDataSet)
    if similar_resumes:
        return jsonify({"results": similar_resumes})
    return jsonify({"results": [], "message": "No matching resumes found."})

@app.route("/voice_query", methods=["POST"])
def voice_query():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening for voice input...")
            audio = recognizer.listen(source)
            query = recognizer.recognize_google(audio)
            print(f"Voice query received: {query}")

        # Find the top 5 similar resumes based on the query
        similar_resumes = find_similar_resumes(query, tfidf_matrix, resumeDataSet)

        if similar_resumes:
            return jsonify({"query": query, "results": similar_resumes})
        return jsonify({"query": query, "message": "No matching resumes found."})

    except sr.UnknownValueError:
        return jsonify({"error": "Sorry, I couldn't understand the audio. Please try again."})
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"})
    except Exception as e:
        return jsonify({"error": f"Error processing voice input: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
