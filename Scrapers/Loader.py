import time
import json
import os
import glob
import re
import requests
import traceback
import logging
from ast import Return
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
from urllib.parse import urljoin
from datetime import datetime
from newspaper import Article
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


#Defines the logging function ,or how logs will be displayed 
logging.basicConfig(
    filename='Logs/Loader.log',
    level=logging.INFO,  # Change to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Accept-Encoding": "gzip, deflate, br",
}  #Mimics real browser request


def clear_old_screenshots(log_dir="Logs"): #Function to delete screenshots
    if not os.path.exists(log_dir):
        return  # Nothing to clear
    for file in glob.glob(os.path.join(log_dir, "*_fail.png")): #If file exists
        try:
            os.remove(file) #Removes file
            print(f"Deleted old screenshot: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")


#A function to get advanced logs by accessing the actual url 
def advanced_get(session, base_url, relative_url):
    try:
        full_url = urljoin(base_url, relative_url)#Base url accesses parent site , relative url accesses the exact site driver sees
        logging.info(f"Attempting to fetch: {full_url}") #Fetches the site driver sees

        response = session.get(full_url, headers=headers, timeout=10, allow_redirects=True) #Creates a link between site and program to send messages across

        # Log basic response info
        logging.info(f"[{response.status_code}] {full_url} (Final URL: {response.url})")
        logging.debug(f"Headers: {response.headers}")
        logging.debug(f"Content Preview: {response.text[:500]}")  # log first 500 characters
        logging.debug(traceback.format_exc())

        # Check if it's HTML
        if "text/html" not in response.headers.get("Content-Type", ""):
            logging.warning(f"Non-HTML content at {full_url}")

        response.raise_for_status()
        return response.text #Returns the messages from the site

    #Log messages
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error for {full_url}: {e.response.status_code} - {e.response.reason}")
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection Error for {full_url}: {str(e)}")
    except requests.exceptions.Timeout:
        logging.error(f"Timeout occurred while fetching {full_url}")
    except Exception as e:
        logging.error(f"Unhandled error for {full_url}: {str(e)}")
        logging.debug(traceback.format_exc())

    return None

with open('Datasets/miniature.json', 'r') as f: #Loads json file 
    article = json.load(f) 
    json_dict=article["articles"] #articles is dictionary inside the json file ,this loads the actual articles into json_dict

    os.makedirs('BERT_Content', exist_ok=True)  # Ensure directory exists

def setup_driver(): #Setup Chrome WebDriver
    chrome_options=Options()
    chrome_options.add_argument("--headless")  # Run without GUI
    chrome_options.add_argument("--no-sandbox") # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options) #Creates web driver instance

def sanitize_filename(name):
    # Remove any character that's not alphanumeric, dash, underscore, or space
    name = re.sub(r'[^\w\s-]', '', name)
    # Replace whitespace/newlines with underscore
    name = re.sub(r'[\s]+', '_', name)
    return name[:150] 

def click_and_read(driver):
    try:
        buttons=driver.find_elements(By.XPATH, "//button[contains(text(),'Read More') or contains(text(),'Continue Reading') or contains(text(),'Show more')]")#Finds button inside of the html
        for btn in buttons:
            if btn.is_displayed() and btn.is_enabled():#Checks if button is enabled 
                btn.click() 
                time.sleep(3) #Waits for dynamic content to load
                break
    except Exception as e:
        logging.error(f"Failed to click read more :{e}") #Log message
        logging.debug(traceback.format_exc())
        
def scroll_and_extract(driver, article, max_scrolls=10): #Function to scroll through page and extract all the content
    title = article.get('title')
    link = article.get('link')

    if not title or not link:
        logging.info("Missing title or link.") #Log message
        logging.debug(traceback.format_exc())
        return None

    try:
        driver.get(link)
    except (TimeoutException, WebDriverException):
        logging.info(f"Failed to load: {link}") #Log message
        logging.debug(traceback.format_exc())
        return None

    time.sleep(3)
    last_height = driver.execute_script("return document.body.scrollHeight") #Gets Height of the page to scroll properly
    full_content="" #Initialize empty string to constantly append content through on every scroll
    seen_texts=set() #Initializes string to save text that is already seen

    # Scroll to load dynamic content
    for _ in range(max_scrolls):
        no_of_scrolls=0
        # Wait for a key article content element
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article, div.article__content, div.mntl-sc-page, div.content-container"))
            ) #Waits for certain tags to appear before proceeding 
            for scroll_num in range(no_of_scrolls):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")#Scrolls the page
            click_and_read(driver) 
            time.sleep(2) #Waits for dynamic content to load 
        except Exception as e:
            logging.error(f"Initial scroll failed :{e}") #Log message
            logging.debug(traceback.format_exc())
        # Extract text from page
        try:
            html=advanced_get(requests.Session(),base_url=link,relative_url='') #Displays the html of page after scroll
            article_content = Article(link) #Newspapery3k initializes article class with the link given        
            article_content.set_html(html)  #Newspapery3k collects the html tags
        except Exception as e:
            logging.error(f"Advanced get failed for{link} :{e}")   
            logging.debug(traceback.format_exc())
        try:
            article_content.parse() #Newspapery3k extracts the main content
            content_piece=article_content.text.strip() #Converts text
            if content_piece not in seen_texts:
                full_content+= "+\n" +content_piece #Appends text 
                seen_texts.add(content_piece) #Adds text 
        except Exception as e:
            logging.warning(f"Parse Failed at scroll :{e}")#Log message
            logging.debug(traceback.format_exc())
            
        try:
            new_height = driver.execute_script("return document.body.scrollHeight") #Height of the scrolled page 
            if new_height == last_height: #Height of new scrolled page is same to previous page
                break
            last_height = new_height 
        except Exception as e:
            logging.error(f"Height comparison failed at {last_height} :{e}")
            logging.debug(traceback.format_exc())

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
        
    
def extract_multiple_articles(inner_dict, max_scrolls=10):
    driver = setup_driver()  # Initialize the WebDriver
    all_articles = []
    
    for dict_content in inner_dict: #Iterates through every link in scraped list
        logging.info(f"Extracting article: {dict_content['title']}")
        logging.debug(traceback.format_exc())
        try:
            if not is_browser_alive(driver): #Checks if browser is working
                driver.quit()
                driver = setup_driver()
            data = scroll_and_extract(driver, dict_content, max_scrolls=max_scrolls) #Extracts contents from articles
            all_articles.append(data)
        except:
            logging.error(f"Error extracting article: {dict_content['title']}")
            logging.debug(traceback.format_exc())
            screenshot_path = os.path.join("Logs", f"{sanitize_filename(dict_content['title'])[:40]}_fail.png") #Takes screenshot of the page when failure occured
            driver.save_screenshot(screenshot_path)
            logging.info(f"[SCREENSHOT] Saved failure screenshot to {screenshot_path}")
            continue
    
    driver.quit()    
    return all_articles
    
def is_browser_alive(driver): #Checks if browser is still running
    try:
        driver.title
        return True
    except:
        logging.debug(traceback.format_exc())
        return False
    
    
if __name__=="__main__":
    clear_old_screenshots() #Deletes old screenshot
    extracted_articles = extract_multiple_articles(json_dict, max_scrolls=10) #Calls the complete extraction process
    logging.info(f"Total Extracted articles:{len(extracted_articles)}")
    
    
    
    
        
    
    

