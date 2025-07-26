import re
import asyncio
import logging

from playwright.async_api import async_playwright

# Your custom utility
from utils import save_file,logging

async def scrape_with_timeout(url, title ,timeout=60):
    try:
        await asyncio.wait_for(playwright_article_text(url, title), timeout=timeout)#Sets a timeout for the scraping operation
    except asyncio.TimeoutError:
        logging.error(f"Timeout while scraping {url} after {timeout}s.") #Log message
    except Exception as e:
        logging.error(f"Unhandled exception during scraping {url}: {e}", exc_info=True) #Log message

# Wrapper for synchronous use
def get_article_text_playwright(url, title,timeout=60):
    return asyncio.run(scrape_with_timeout(url, title,timeout)) #Async wrapper to run async functions

# Async article text extractor
async def playwright_article_text(url, title):
    logging.info(f"Starting Playwright scraping for: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True) #Launches headless browser
            page = await browser.new_page() #Creates a new page on browser

            try:
                await page.goto(url, timeout=10000, wait_until="domcontentloaded") #Waits for page to load
            except TimeoutError:
                logging.warning(f"Timeout at 10s, retrying with 30s...")
                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded") #Retries the page with a longer timeout
                except Exception as e:
                    logging.error(f"Failed even after retry: {e}")

                logging.error(f"Error loading page: {e}")
                await browser.close() #Closes the browser
                return

            # 1. Grab article or body text
            try:
                body_text = await page.locator("article").inner_text() #Locates article tag  and extracts the text inside it
                logging.info("Extracted text from <article>.") #Log message
            except Exception:
                try:
                    body_text = await page.locator("body").inner_text() #Fallsback to body and extracts the text inside it
                    logging.warning("Fell back to extracting text from <body>.") #Log message
                except Exception as e:
                    logging.error(f"Failed to extract any main text: {e}") #Log message
                    body_text = "" #Resets body text if extraction fails

            # 2. Heuristic extracts long quoted sections from <div> tags
            extracted_script_texts = [] #Initializes a list to store extracted texts
            try:
                script_tags = await page.locator("div").all() #Locates all div tags 
                logging.info(f"Found {len(script_tags)} <div> tags to scan.")

                for tag in script_tags:
                    try:
                        content = await tag.text_content() #Checks text inside every div tag
                        if content and any(x in content for x in ['article', 'body', 'content', 'strong']): #Checks these particular tags inside div
                            quotes = re.findall(r'"(.*?)"', content, re.DOTALL) #Extracts all text inside the above tags within the div
                            long_quotes = [q for q in quotes if len(q) > 200] #Filters out long quotes
                            extracted_script_texts.extend(long_quotes) #Adds the long quotes to the list
                    except Exception:
                        continue
            except Exception as e:
                logging.error(f"Error while parsing <div> tags: {e}")

            await browser.close()

            # Selects either the body text or the text inside the div tags
            if extracted_script_texts:
                final_content = "\n\n".join(dict.fromkeys(extracted_script_texts[:3]))  # Remove duplicates and takes content inside divs as final
                logging.info(f"Extracted {len(extracted_script_texts[:3])} large quoted sections.")
            elif body_text and len(body_text) > 500:
                final_content = body_text # Selects body text if it is long enough
                logging.info("Used fallback body text.")
            else:
                final_content = "[Article text not found]"
                logging.warning("No usable content found.")

            save_file(title, final_content) #Saves the content to a file 
            logging.info(f"Saved article to file: {title}")
    except Exception as e:
        logging.critical(f"Unhandled exception in Playwright article fetcher: {e}", exc_info=True)
        