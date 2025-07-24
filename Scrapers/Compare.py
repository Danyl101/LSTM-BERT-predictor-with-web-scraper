import os
import json
import shutil
import logging
import re
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(
    filename='Logs/Compare.log',
    level=logging.INFO,  # Change to DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# This gets the absolute path of the directory that this script is in
SCRIPT_DIR = Path(__file__).resolve().parent

# This goes up to the "Prediction Model" folder (one level up from Scrapers)
BASE_DIR = SCRIPT_DIR.parent

def title_to_filename(title):
    return title.strip().replace("/", "_") + ".txt"

def word_count(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return len(f.read().split())

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
os.makedirs('Final_bert_folder',exist_ok=True)


# === CONFIG ===
articles_json_path = Path("Datasets/processed_articles.json")
scraped_json_path = Path("Datasets/scraped_articles.json")
base_dir = Path(__file__).resolve().parent
bert_folder =BASE_DIR/"BERT_Content"
final_bert_folder = Path("Final_bert_folder")

# Ensure output folder exists
final_bert_folder.mkdir(parents=True, exist_ok=True)

# === Load Data ===
scraped_dict_outside = load_json(scraped_json_path)
scraped_dict=scraped_dict_outside['articles']

articles_json = []
remaining_articles = []

print(f"[INFO] Checking {len(scraped_dict)} articles...")

for article in tqdm(scraped_dict):
    title = article.get("title")
    if not title:
        logging.warning("Article without title skipped.")
        continue

    filename = title_to_filename(title)
    filepath = bert_folder / filename
    print(filepath)
    logging.debug(f"Checking file: {filepath}")

    if filepath.exists():
        try:
            wc = word_count(filepath)
            logging.debug(f"Word count for {filename}: {wc}")
        except Exception as e:
            logging.error(f"Failed to read file {filepath}: {e}")
            remaining_articles.append(article)
            continue

        if wc >= 150:
            try:
                shutil.move(str(filepath), final_bert_folder / filename)
                logging.info(f"Moved: {filename}")
                articles_json.append(article)
            except Exception as e:
                logging.error(f"Error moving {filename}: {e}")
                remaining_articles.append(article)
        else:
            logging.info(f"Skipped (word count too low): {filename}")
            remaining_articles.append(article)
    else:
        logging.info(f"Skipped (file not found): {filename}")
        remaining_articles.append(article)

# === Save Updated Files ===
save_json(articles_json_path, articles_json)
save_json(scraped_json_path, remaining_articles)

print(f"[DONE] Moved {len(articles_json)} completed articles to final folder.")
print(f"[REMAINING] {len(remaining_articles)} articles left to process.")



