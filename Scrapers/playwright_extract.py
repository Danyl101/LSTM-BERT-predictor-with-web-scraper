import re
import asyncio
import logging

from playwright.async_api import async_playwright

# Your custom utility
from utils import save_file,logging


# Wrapper for synchronous use
def get_article_text_playwright(url, title):
    return asyncio.run(playwright_article_text(url, title))

# Async article text extractor
async def playwright_article_text(url, title):
    logging.info(f"Starting Playwright scraping for: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                await page.goto(url, timeout=10000, wait_until="domcontentloaded")
            except TimeoutError:
                logging.warning(f"Timeout at 10s, retrying with 30s...")
                try:
                    await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                except Exception as e:
                    logging.error(f"Failed even after retry: {e}")

                logging.error(f"Error loading page: {e}")
                await browser.close()
                return

            # 1. Grab article or body text
            try:
                body_text = await page.locator("article").inner_text()
                logging.info("Extracted text from <article>.")
            except Exception:
                try:
                    body_text = await page.locator("body").inner_text()
                    logging.warning("Fell back to extracting text from <body>.")
                except Exception as e:
                    logging.error(f"Failed to extract any main text: {e}")
                    body_text = ""

            # 2. Extract from script-like <div> blocks
            extracted_script_texts = []
            try:
                script_tags = await page.locator("div").all()
                logging.info(f"Found {len(script_tags)} <div> tags to scan.")

                for tag in script_tags:
                    try:
                        content = await tag.text_content()
                        if content and any(x in content for x in ['article', 'body', 'content', 'strong']):
                            quotes = re.findall(r'"(.*?)"', content, re.DOTALL)
                            long_quotes = [q for q in quotes if len(q) > 200]
                            extracted_script_texts.extend(long_quotes)
                    except Exception:
                        continue
            except Exception as e:
                logging.error(f"Error while parsing <div> tags: {e}")

            await browser.close()

            # Final content selection
            if extracted_script_texts:
                final_content = "\n\n".join(dict.fromkeys(extracted_script_texts[:3]))  # Remove duplicates
                logging.info(f"Extracted {len(extracted_script_texts[:3])} large quoted sections.")
            elif body_text and len(body_text) > 500:
                final_content = body_text
                logging.info("Used fallback body text.")
            else:
                final_content = "[Article text not found]"
                logging.warning("No usable content found.")

            save_file(title, final_content)
            logging.info(f"Saved article to file: {title}")
    except Exception as e:
        logging.critical(f"Unhandled exception in Playwright article fetcher: {e}", exc_info=True)
        