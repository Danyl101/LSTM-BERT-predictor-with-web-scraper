import logging
import re
import traceback
import os


headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Accept-Encoding": "gzip, deflate, br",
}  #Mimics real browser request

#Defines the logging function ,or how logs will be displayed 
logging.basicConfig(
    filename='Logs/Loader.log',
    level=logging.INFO,  # Change to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def sanitize_filename(name):
    # Remove any character that's not alphanumeric, dash, underscore, or space
    name = re.sub(r'[^\w\s-]', '', name)
    # Replace whitespace/newlines with underscore
    name = re.sub(r'[\s]+', '_', name)
    return name[:150] 
    
def save_file(title, full_content):
    try:
        if not  full_content.strip():
            logging.info(f"No content found for: {title}") #Log message
            logging.debug(traceback.format_exc())
            return None
    except Exception as e:
        logging.error(f"Content cleaning failed : {e}")
        logging.debug(traceback.format_exc())

    try:
        # Save as .txt file
        filename = sanitize_filename(title) + ".txt" #Cleans file names
        filepath = os.path.join("BERT_Content", filename) #Creates filepath for the files
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content) #Writes content to the file

        logging.info(f"Saved: {filename}") #Log message
        return full_content
    except Exception as e:
        logging.error(f"Error saving file {filename} :{e}")