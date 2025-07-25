import os
import json
import re

MINIATURE_DIR = "Datasets"
BERT_CONTENT_DIR = "BERT_Content"

def clean_string(s):
    """Clean and normalize the title/filename."""
    s = s.replace('\n', ' ')        # Replace newline with space
    s = re.sub(r'[^\w\s]', '', s)   # Remove special characters
    s = s.strip().lower()           # Lowercase and trim
    s = re.sub(r'\s+', '_', s)      # Convert spaces to underscores
    return s


# Collect all filenames in BERT_Content (normalized)
bert_filenames = [clean_string(os.path.splitext(f)[0]) for f in os.listdir(BERT_CONTENT_DIR)]

for filename in os.listdir(MINIATURE_DIR):
    if filename.endswith("miniature.json"):
        path = os.path.join(MINIATURE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                access_data = data['articles']
                articles = []  # ✅ Reset per file
                for content in access_data:
                    title = content['title']
                    link = content['link']
                    cleaned_title = clean_string(title)

                    if cleaned_title in bert_filenames:
                        articles.append({'title': cleaned_title, 'link': link})
                    # optional: store unmatched if needed
            except Exception as e:
                print(f"[Error reading {filename}] {e}")

# Save the final result
with open("Datasets/updated_article_list.json", "w", encoding="utf-8") as f:
    json.dump({"articles":articles}, f, indent=2, ensure_ascii=False)

print("✅ Mapping complete. Output saved to title_to_filename_mapping.json")
