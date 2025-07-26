import os
import time
import random
import logging
import traceback
import requests

from urllib.parse import urljoin
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from newspaper import Article

# Assuming these are defined in your own codebase
from utils import sanitize_filename, save_file,logging,headers


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

def click_and_read(driver):
    try:
        buttons=driver.find_elements(By.XPATH, "//button[contains(text(),'Read More') or contains(text(),'Continue Reading') or contains(text(),'Show more')]")#Finds button inside of the html
        for btn in buttons:
            time.sleep(random.uniform(2, 5))
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

    full_content="" #Initialize empty string to constantly append content through on every scroll
    seen_texts=set() #Initializes string to save text that is already seen
    time.sleep(3)
    last_height = driver.execute_script("return document.body.scrollHeight")#Gets Height of the page to scroll properly
    
    # Scroll to load dynamic content
    for _ in range(max_scrolls):
        time.sleep(random.uniform(2, 4))
        no_of_scrolls=0
        # Wait for a key article content element
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "article, div.article__content, div.mntl-sc-page, div.content-container"))
            ) #Waits for certain tags to appear before proceeding 
            for num_scrolls in range(no_of_scrolls):
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
            article_content.set_html(html)#Newspapery3k collects the html tags
        except Exception as e:
            logging.error(f"Advanced get failed for{link} :{e}")   
            logging.debug(traceback.format_exc())
        try:
            article_content.parse() #Newspapery3k extracts the main content    
            content_piece=article_content.text.strip() #Converts text
            if content_piece not in seen_texts:
                full_content+= "+\n" +content_piece #Appends text 
                seen_texts.add(content_piece) #Adds text
                screenshot_path = os.path.join("Logs", f"{sanitize_filename(title)[:40]}_fail.png") #Takes screenshot of the page when failure occured
                driver.save_screenshot(screenshot_path)
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
        
    save_file(title, full_content) #Saves the file after every scroll