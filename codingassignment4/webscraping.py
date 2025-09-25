from openai import OpenAI
import os
from bs4 import BeautifulSoup
import requests
import json
from supabase import create_client, Client
import csv
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# llm client
ENDPOINT = os.environ.get("ENDPOINT")
API_KEY = os.environ.get("API_KEY")
deployment_name = "gpt-4o"
client = OpenAI(
    base_url=ENDPOINT,
    api_key=API_KEY
)

# structurer
url = "https://www.lovecrafts.com/en-us/l/crochet/crochet-patterns/free-crochet-patterns?srsltid=AfmBOoroxO51P0JvKgR46Okr3Uyx_XitLiXFHusK4sBF2AX_1TLRzqho"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

grid_items = soup.find_all('li', class_='products__grid-item')

patterns_text = ""
for i in grid_items:
    patterns_text += i.get_text(separator=" ", strip=True) + "\n"

# Save the extracted data to a file in JSON format
with open("textblob.txt", "w") as file:
    file.write(patterns_text)

# Read the file content
with open("textblob.txt", "r") as file:
    content = file.read()

response = client.chat.completions.create(
    model=deployment_name,
    messages=[
        {
            "role": "developer",
            "content": "Return in JSON. Structure in an array of objects with keys: pattern_name, subtitle, price. Ensure that the JSON is properly formatted and parsable."
        },
        {
            "role": "user",
            "content": f"Based on the {{content}}, can you extract the pattern name, subtitle, price and store it in JSON format? Here is the html_content: {content}"
        }
    ],
    response_format={"type": "json_object"}
)

print(response.choices[0].message.content)

# loader
json_content = response.choices[0].message.content
json_data = json.loads(json_content)

timestamp = datetime.now().isoformat()

patterns = json_data["patterns"]

for pattern in patterns:
    pattern["updated_at"] = timestamp

with open("crochet_patterns.csv", "w", newline='') as f:
    cw = csv.writer(f)
    header = ["id"] + list(patterns[0].keys())
    cw.writerow(header)

    for idx, pattern in enumerate(patterns):
        row = [idx + 1] + list(pattern.values())
        cw.writerow(row)

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

csv = pd.read_csv("crochet_patterns.csv")

table_name = "crochetpattern"
response = supabase.table(table_name).upsert(csv.to_dict(orient="records")).execute()