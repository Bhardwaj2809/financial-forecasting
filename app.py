from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import docx

# Initialize Flask app
app = Flask(__name__)

# Folder to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# Function to check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Read file content (simple example)
            content = ""
            if filename.endswith(".pdf"):
                reader = PdfReader(filepath)
                for page in reader.pages:
                    content += page.extract_text() + "\n"
            elif filename.endswith(".docx"):
                doc = docx.Document(filepath)
                for para in doc.paragraphs:
                    content += para.text + "\n"

            # Here you can add your predictive analysis code
            # For now, just returning first 500 characters
            preview = content[:500] + ("..." if len(content) > 500 else "")
            return f"<h3>File uploaded successfully!</h3><pre>{preview}</pre>"

        else:
            return "Invalid file type. Only PDF and DOCX are allowed."

    # GET request
    return render_template("upload.html")


if __name__ == "__main__":
    # Debug mode on for auto reload
    app.run(debug=True)
