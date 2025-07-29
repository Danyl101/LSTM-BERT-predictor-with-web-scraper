import time
import json
import logging
import traceback
from httpx import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
import requests
import urllib.robotparser
from urllib.parse import urlparse

logging.basicConfig(
    filename='Logs/Scraper.log',
    level=logging.INFO,  # Change to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def setup_driver(): #Setup Chrome WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run without GUI
    chrome_options.add_argument("--no-sandbox") # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) #Creates web driver instance

def fetch_robots_txt(site):
    try:
        resp = requests.get(site + "/robots.txt", timeout=10)
        return resp.text
    except:
        return None
    
def can_scrape(url, user_agent='*'):
    try:
        # Extract domain
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Construct robots.txt URL
        robots_url = f"{base_url}/robots.txt"

        # Read and parse robots.txt
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()

        # Check if allowed
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        logging.error(f"[WARN] Could not parse robots.txt for {url}: {e}")
        # Assume allowed if robots.txt is unreachable
        return True
    
    #5542

def scroll_and_scrape(driver, url, max_scrolls=10):#Function to scroll and scrape articles from a given URL
    try:
        driver.get(url) 
    except TimeoutException: #Handles site timeouts 
        logging.error(f"Timeout while loading {url}, retrying...")
        return []
      
    time.sleep(3) #Wait for dyanmic content to load 

    last_height = driver.execute_script("return document.body.scrollHeight") #Gets Height of the page to scroll properly
    raw_articles =[]

    for _ in range(max_scrolls): #Function to Scroll through page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") #Scrolls through page
        time.sleep(2) #Waits for dynamic content to load 

        try:
            elems = driver.find_elements(By.TAG_NAME, 'a') #Acquires all anchor tags
            for elem in elems: #Iterates through elements 
                try:
                    link = elem.get_attribute('href') #Acquires link from these elements 
                    title = elem.text.strip() #Produces title from link
                    if link and title and len(title) > 20: # Filters out garbage links like javascript(0)
                        raw_articles.append((title.lower(), link.lower())) 
                except StaleElementReferenceException: #Handles stale DOM elements
                    continue        
        except Exception:
            logging.error("Error while fetching elements, retrying...")
            continue
            
        new_height = driver.execute_script("return document.body.scrollHeight") #Height of the scrolled page 
        if new_height == last_height: #Height of new scrolled page is same to previous page
            break
        last_height = new_height 
        
        cleaned_articles = []
        goodlist = [
    "market", "finance", "stock", "business", "economy", "invest", "earnings",
    "ipo", "valuation", "bank", "crypto", "nifty", "sensex", "index",
    "inflation", "budget", "gdp", "fii", "dii", "commodity", "forex", "mutual"]# Clean Sites and links 
        

        for title, link in raw_articles: #Function to filter only good sites from the articles extracted 
            if any(good in link.lower() for good in goodlist):
                if(can_scrape(link)):
                    cleaned_articles.append({
                        "title": title.lower(),
                        "link": link.lower()
                }) #Appends clean sites to final list 

    return cleaned_articles #Returns titles and links of articles 

def scrape_multiple_sites(sites, max_scrolls=10): #Function to access multiple sites 
    driver = setup_driver() #Initializes web driver 
    all_data = []

    for site in sites:
        logging.info(f"Scraping {site}...")
        try:
            if not is_browser_alive(driver): #Checks if web driver is dead or alive
                driver.quit() #Deletes existing driver 
                driver = setup_driver() #Initializes new driver
            data = scroll_and_scrape(driver, site, max_scrolls=max_scrolls) #Calls scroll and scrape function
            all_data.extend(data)
        except Exception as e:
            logging.error(f"Error scraping {site}: {e}")
            continue

    driver.quit()
    return all_data

def is_browser_alive(driver):#Checks if a driver is healthy or not 
    try:
        driver.title  # Will raise error if session is invalid
        return True
    except:
        return False


if __name__ == "__main__":
    websites = [
    # 🇮🇳 India
    "https://economictimes.indiatimes.com/news",
    "https://economictimes.indiatimes.com/markets",
    "https://www.moneycontrol.com/news/business",
    "https://www.moneycontrol.com/news/market",
    "https://www.business-standard.com/category/markets",
    "https://www.financialexpress.com/market",
    "https://www.cnbctv18.com/market",
    "https://www.ndtvprofit.com",
    "https://www.thehindubusinessline.com/markets",
    "https://www.indiainfoline.com/news",  # stock/IPO updates

    # 🌐 United States / Global
    "https://finance.yahoo.com",
    "https://www.investing.com/news/stock-market-news",
    "https://www.nasdaq.com/news-and-insights",
    "https://www.bloomberg.com/markets",
    "https://www.reuters.com/markets",
    "https://www.marketwatch.com/latest-news",
    "https://www.benzinga.com/news",
    "https://www.zacks.com/stock/news",
    "https://www.seekingalpha.com/market-news",
    "https://www.fool.com/investing",
    "https://www.wsj.com/news/markets",  # Wall Street Journal (paywalled)
    "https://www.barrons.com/market-data",  # Market-focused news
    "https://www.theinvestorschronicle.co.uk/",  # London-based weekly by FT

    # 🌍 Canada
    "https://www.bnnbloomberg.ca/",  # BNN Bloomberg

    # 🇦🇺 Australia / NZ
    "https://www.afr.com/",  # Australian Financial Review
    "https://business.nine.com.au/business-news",  # general market
    "https://www.businessspectator.com.au/",  # Australian business insights
    "https://www.capitalbrief.com/",  # Australian business summary
    "https://www.nzherald.co.nz/business/",  # NZ business section

    # 🇬🇧 United Kingdom / Europe
    "https://www.ft.com/markets",  # Financial Times :contentReference[oaicite:1]{index=1}
    "https://www.moneyweek.com/",  # British investment weekly :contentReference[oaicite:2]{index=2}
    "https://www.reuters.com/finance",  # Reuters finance hub :contentReference[oaicite:3]{index=3}

    # 🌏 Hong Kong / Asia-Pacific
    "https://www.financeasia.com/",  # Asia‑Pacific capital markets :contentReference[oaicite:4]{index=4}

    # 💱 Forex / Macro-focused
    "https://www.fxstreet.com/news",
    "https://www.tradingeconomics.com/news",  # macro indicators & policy

    # 🪙 Crypto (if you pivot later)
    "https://www.coindesk.com/",
    "https://www.cointelegraph.com/",

    # 🧠 General / Quant Education
    "https://www.investopedia.com/news/",
]
    #4

    cleaned_articles = scrape_multiple_sites(websites, max_scrolls=15)
    logging.info(f"Total articles scraped: {len(cleaned_articles)}") #Returns no of articles that are available after cleaning

    with open("Datasets/scraped_articles.json", "w", encoding='utf-8') as f: #Writes the processed data into a json file containing links and titles 
        json.dump({"articles": cleaned_articles}, f, indent=2, ensure_ascii=False) 


                    