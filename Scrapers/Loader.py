import time
import json
import os
import re
import random
import traceback
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

from playwright_extract import get_article_text_playwright  # Importing the Playwright function

from Selenium_newspaper import scroll_and_extract  # Importing the Selenium function


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

def is_browser_alive(driver): #Checks if browser is still running
    try:
        driver.title
        return True
    except:
        logging.debug(traceback.format_exc())
        return False
   
def extract_multiple_articles(inner_dict, max_scrolls=10):
    driver = setup_driver()  # Initialize the WebDriver
    all_articles = []
    
    for dict_content in inner_dict: #Iterates through every link in scraped list
        time.sleep(random.uniform(2, 5))
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
            continue
    
    driver.quit()    
    return all_articles
    
if __name__=="__main__":
    extracted_articles = extract_multiple_articles(json_dict, max_scrolls=10) #Calls the complete extraction process
    logging.info(f"Total Extracted articles:{len(extracted_articles)}")
    
    
    
    
        
    
    

