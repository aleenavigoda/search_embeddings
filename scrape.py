import os
import json
import time
import re
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Initialize Supabase client
url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase: Client = create_client(url, key)

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_metadata(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.title.string if soup.title else None
        if not title:
            h1 = soup.find('h1')
            title = h1.text.strip() if h1 else None

        meta_description = soup.find('meta', attrs={'name': 'description'})
        description = meta_description['content'] if meta_description else None
        if not description:
            first_p = soup.find('p')
            description = first_p.text.strip()[:200] + '...' if first_p else None

        return clean_title(title), description
    except Exception as e:
        print(f"Error fetching metadata for {url}: {str(e)}")
        return None, None

def clean_title(title):
    if not title:
        return None
    
    suffixes = ['| Age of Invention', '| Substack']
    for suffix in suffixes:
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()
    
    return title.strip()

def get_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

def get_tags(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates topic tags for essays."},
                    {"role": "user", "content": f"Generate a list of 5-10 topic tags for the following text. Return only the list of tags in JSON array format:\n\n{text}"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            tags = json.loads(content)
            if isinstance(tags, list):
                return tags
            else:
                print(f"Unexpected tags format: {tags}")
                return []
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error getting tags, retrying... (Attempt {attempt + 1})")
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to get tags after {max_retries} attempts: {str(e)}")
                return []

def update_new_rows():
    # Fetch rows from the urls_table where any of title, description, smol_embedding, or tags are null
    response = supabase.table("urls_table").select("*").or_("title.is.null,description.is.null,smol_embedding.is.null,tags.is.null").execute()
    
    rows = response.data
    print(f"Processing {len(rows)} new rows.")
    
    for row in tqdm(rows, desc="Processing new rows"):
        url_id = row['id']
        url = row['url']
        
        update_data = {}
        
        # Fetch metadata if title or description is null
        if row['title'] is None or row['description'] is None:
            title, description = get_metadata(url)
            if title and row['title'] is None:
                update_data['title'] = title
            if description and row['description'] is None:
                update_data['description'] = description
        
        # Use existing or new title and description for embedding and tags
        title = update_data.get('title', row['title']) or ""
        description = update_data.get('description', row['description']) or ""
        combined_text = f"{title}\n{description}\n{url}"
        
        # Generate embedding if null
        if row['smol_embedding'] is None:
            embedding = get_embedding(combined_text)
            update_data['smol_embedding'] = embedding
        
        # Generate tags if null
        if row['tags'] is None:
            tags = get_tags(combined_text)
            update_data['tags'] = tags
        
        # Update the row if there's new data
        if update_data:
            try:
                supabase.table("urls_table").update(update_data).eq("id", url_id).execute()
                print(f"Updated data for id {url_id}")
            except Exception as e:
                print(f"Error updating row {url_id}: {str(e)}")
        
        # Sleep to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    update_new_rows()