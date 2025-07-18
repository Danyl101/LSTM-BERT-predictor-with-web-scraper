import time
import json
from httpx import TimeoutException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException

def setup_driver(): #Setup Chrome WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run without GUI
    chrome_options.add_argument("--no-sandbox") # Bypass OS security model
    chrome_options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options) #Creates web driver instance

def scroll_and_scrape(driver, url, max_scrolls=10):#Function to scroll and scrape articles from a given URL
    try:
        driver.get(url) 
    except TimeoutException: #Handles site timeouts 
        print(f"Timeout while loading {url}, retrying...")
        return []
      
    time.sleep(3) #Wait for dyanmic content to load 

    last_height = driver.execute_script("return document.body.scrollHeight") #Gets Height of the page to scroll properly
    raw_articles =[]
    cleaned_articles=[]

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
            print("Error while fetching elements, retrying...")
            continue
            
        new_height = driver.execute_script("return document.body.scrollHeight") #Height of the scrolled page 
        if new_height == last_height: #Height of new scrolled page is same to previous page
            break
        last_height = new_height 
        
        cleaned_articles = []
        blacklist = [
            "login", "signup", "contact", "terms", "privacy", "register", "feedback", "ads", "help",
            "sports", "cricket", "entertainment", "movies", "lifestyle", "tech", "gadgets", "education", 
            "opinion", "horoscope", "health", "travel", "fashion", "food", "video"
            ] #Blacklisted Sites and links 

        
        for title, link in raw_articles: #Function to filter out blacklisted sites from the articles extracted 
            if not any(bad in link.lower() for bad in blacklist):
                cleaned_articles.append({
                    "title": title.lower(),
                    "link": link.lower()
                }) #Appends clean sites to final list 

    return [{"title": t, "link": l} for t, l in cleaned_articles] #Returns titles and links of articles 

def scrape_multiple_sites(sites, max_scrolls=10): #Function to access multiple sites 
    driver = setup_driver() #Initializes web driver 
    all_data = []

    for site in sites:
        print(f"Scraping {site}...")
        try:
            if not is_browser_alive(driver): #Checks if web driver is dead or alive
                driver.quit() #Deletes existing driver 
                driver = setup_driver() #Initializes new driver
            data = scroll_and_scrape(driver, site, max_scrolls=max_scrolls) #Calls scroll and scrape function
            all_data.extend(data)
        except Exception as e:
            print(f"Error scraping {site}: {e}")
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
    "https://economictimes.indiatimes.com/news",
    "https://economictimes.indiatimes.com/markets",  # Market-specific
    "https://www.livemint.com/latest-news",
    "https://www.livemint.com/market",  # Stocks, IPOs, analysis
    "https://www.moneycontrol.com/news/business",
    "https://www.moneycontrol.com/news/market",  # Markets tab
    "https://www.business-standard.com/category/markets",  # Equities, Commodities, Forex
    "https://www.financialexpress.com/market",  # Covers stocks, bonds, IPOs
    "https://www.nasdaq.com/news-and-insights",  # Global + US market focus
    "https://www.investing.com/news/stock-market-news",  # Clean global stock news
    "https://www.bloomberg.com/markets",  # Premier financial news
    "https://www.reuters.com/markets",  # Factual, global market data
    "https://www.cnbctv18.com/market",  # Indian stocks, sensex, nifty
    "https://www.ndtvprofit.com",  # NDTV's business/market section
    "https://www.tradingview.com/news",  # Market sentiment + charts + opinions
]


    cleaned_articles = scrape_multiple_sites(websites, max_scrolls=15)
    print(f"Total articles scraped: {len(cleaned_articles)}") #Returns no of articles that are available after cleaning

    with open("Datasets/scraped_articles.json", "w", encoding='utf-8') as f: #Writes the processed data into a json file containing links and titles 
        json.dump({"articles": cleaned_articles}, f, indent=2, ensure_ascii=False) 


                    