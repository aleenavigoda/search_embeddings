from supabase import create_client, Client
from goose3 import Goose

url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"

# Function to connect to Supabase
def connect_to_supabase():
    supabase: Client = create_client(url, key)
    return supabase

# Function to scrape content using Goose3
def scrape_content(url):
    g = Goose()
    try:
        article = g.extract(url=url)
        return article.cleaned_text
    except Exception as e:
        print(f"Failed to extract content for URL {url}: {e}")
        return None

# Function to get URLs from Supabase
def get_urls_from_supabase(supabase):
    try:
        # Fetch all the URLs from the 'urls_table'
        data = supabase.table('urls_table').select('id', 'url').execute()
        return data.data  # This returns a list of dictionaries
    except Exception as e:
        print(f"Failed to fetch URLs: {e}")
        return []

# Function to update the full_text column in Supabase
def update_full_text_in_supabase(supabase, record_id, full_text):
    try:
        # Update the 'full_text' column for the corresponding record
        supabase.table('urls_table').update({'full_text': full_text}).eq('id', record_id).execute()
    except Exception as e:
        print(f"Failed to update full_text for ID {record_id}: {e}")

# Main function to scrape essays and update Supabase
def main():
    supabase = connect_to_supabase()  # Initialize Supabase with provided URL and API key
    urls_records = get_urls_from_supabase(supabase)

    for record in urls_records:
        record_id = record['id']
        url = record['url']
        print(f"Processing URL {url}")

        # Scrape content
        full_text = scrape_content(url)

        if full_text:
            # Update the full_text column in Supabase
            update_full_text_in_supabase(supabase, record_id, full_text)
            print(f"Successfully updated full_text for record ID {record_id}")
        else:
            print(f"Skipping update for record ID {record_id}")

# Ensure the main function is called correctly
if __name__ == "__main__":
    main()