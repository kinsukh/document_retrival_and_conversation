import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Supported file types
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

class DocumentHandler:
    def __init__(self, upload_folder: str):
        """Initialize document handler with upload folder path"""
        self.upload_folder = upload_folder
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file type is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def save_file(self, file) -> Tuple[bool, str]:
        """Save uploaded file to disk"""
        try:
            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.upload_folder, filename)
                file.save(filepath)
                return True, filepath
            return False, "Invalid file type"
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return False, str(e)
    
    def extract_text_from_pdf(self, filepath: str) -> Optional[str]:
        """Extract text content from PDF file"""
        try:
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None

    def extract_text_from_docx(self, filepath: str) -> Optional[str]:
        """Extract text content from DOCX file"""
        try:
            doc = docx.Document(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return None

    def extract_text_from_txt(self, filepath: str) -> Optional[str]:
        """Extract text content from TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            return None

    def process_document(self, filepath: str) -> Tuple[bool, str, Optional[str]]:
        """Process document and extract text based on file type"""
        try:
            file_extension = filepath.rsplit('.', 1)[1].lower()
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(filepath)
            elif file_extension in ['docx', 'doc']:
                text = self.extract_text_from_docx(filepath)
            elif file_extension == 'txt':
                text = self.extract_text_from_txt(filepath)
            else:
                return False, "Unsupported file type", None

            if text is None:
                return False, "Failed to extract text", None
                
            return True, "Text extracted successfully", text
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False, str(e), None

    def cleanup_file(self, filepath: str) -> bool:
        """Remove processed file from disk"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up file: {e}")
            return False
