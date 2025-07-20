from ast import Return
import time
import json
import os
import re
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import WebDriverException
import logging
import traceback
from newspaper import Article

logging.basicConfig(
    filename='Loader.log',
    level=logging.INFO,  # Change to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'
)


with open('Datasets/scraped_articles.json', 'r') as f: #Loads json file 
    article = json.load(f) 
    json_dict=article["articles"] #articles is dictionary inside the json file ,this loads the actual articles into json_dict

    os.makedirs('BERT_Content', exist_ok=True)  # Ensure directory exists

def setup_driver(): #Setup Chrome WebDriver
    chrome_options=Options()
    chrome_options.add_argument("--headless")  # Run without GUI
    chrome_options.add_argument("--no-sandbox") # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=chrome_options) #Creates web driver instance

def sanitize_filename(title): #Function to clean the titles , so that names of files will be clean
    return re.sub(r'[\\/*?:"<>|]', "", title)[:100]

def click_and_read(driver):
    try:
        buttons=driver.find_elements(By.XPATH, "//button[contains(text(),'Read More') or contains(text(),'Continue Reading') or contains(text(),'Show more')]")#Finds button inside of the html
        for btn in buttons:
            if btn.is_displayed() and btn.is_enabled():#Checks if button is enabled 
                btn.click() 
                time.sleep(3) #Waits for dynamic content to load
                break
    except Exception as e:
        logging.error(f"Failed to click read more :{e}")
        
def scroll_and_extract(driver, article, max_scrolls=10): #Function 
    title = article.get('title')
    link = article.get('link')

    if not title or not link:
        logging.info("Missing title or link.")
        return None

    try:
        driver.get(link)
    except (TimeoutException, WebDriverException):
        logging.info(f"Failed to load: {link}")
        return None

    time.sleep(3)
    click_and_read(driver)
    last_height = driver.execute_script("return document.body.scrollHeight") #Gets Height of the page to scroll properly
    full_content=""
    seen_texts=set()

    # Scroll to load dynamic content
    for _ in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    # Extract text from page
        html = driver.page_source  #Displays the html of page after scroll
        article_content = Article(link) #Newspapery3k visits the site 
        article_content.set_html(html)  #Newspapery3k collects the html tags
        
        try:
            article_content.parse()#Newspapery3k extracts the main content
            content_piece=article_content.text.strip()
            if content_piece not in seen_texts:
                full_content+= "+\n" +content_piece
                seen_texts.add(content_piece)
        except Exception as e:
            logging.warning(f"Parse Failed at scroll :{e}")
            continue
        
        new_height = driver.execute_script("return document.body.scrollHeight") #Height of the scrolled page 
        if new_height == last_height: #Height of new scrolled page is same to previous page
            break
        last_height = new_height


    if not  full_content.text.strip():
        logging.info(f"No content found for: {title}")
        return None

    # Save as .txt file
    filename = sanitize_filename(title) + ".txt"
    filepath = os.path.join("BERT_Content", filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(article_content)

    logging.info(f"Saved: {filename}")
    return article_content
    
def extract_multiple_articles(inner_dict, max_scrolls=10):
    driver = setup_driver()  # Initialize the WebDriver
    all_articles = []
    
    for dict_content in inner_dict:
        logging.info(f"Extracting article: {dict_content['title']}")
        try:
            if not is_browser_alive(driver):
                driver.quit()
                driver = setup_driver()
            data = scroll_and_extract(driver, dict_content, max_scrolls=max_scrolls)
            all_articles.append(data)
        except:
            logging.error(f"Error extracting article: {dict_content['title']}")
            continue
    
    driver.quit()    
    return all_articles
    
def is_browser_alive(driver):
    try:
        driver.title
        return True
    except:
        return False
    
if __name__=="__main__":
    extracted_articles = extract_multiple_articles(json_dict, max_scrolls=10)
    (f"Total Extracted articles:{len(extracted_articles)}")
    
    
    
    
        
    
    

