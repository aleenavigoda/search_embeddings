import os
import csv
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from requests.exceptions import RequestException, SSLError
from goose3 import Goose
from goose3.network import NetworkFetcher
from supabase import create_client, Client
import threading

load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

# Global variables
error_code_counter = 1  # Start error code from 1
error_log_file = 'error_log.csv'
success_log_file = 'success_log.csv'
error_code_map = {}  # A map to store error messages and corresponding error codes
error_code_map_lock = threading.Lock()  # Lock to prevent race conditions when updating the error map

# Function to connect to Supabase
def connect_to_supabase():
    supabase: Client = create_client(url, key)
    return supabase

# Function to scrape content using Goose3
def scrape_content(url, retries=1, backoff_factor=2):
    global error_code_counter
    attempt = 0
    delay = 5  # Initial delay in seconds before the first retry

    while attempt <= retries:
        try:
            # Introduce a small random delay between 2 to 5 seconds
            time.sleep(random.uniform(2, 5))
            
            sp_username = os.getenv('SP_USERNAME')
            sp_password = os.getenv('SP_PASSWORD')
            proxy = f"http://{sp_username}:{sp_password}@gate.smartproxy.com:7000"
            print(f"Using Proxy: {proxy}")

            # Create a Goose instance with the custom fetcher
            g = Goose({'network_fetcher': ProxyNetworkFetcher, 'proxy': proxy})
            
            # Extract the article
            article = g.extract(url=url)
            
            if article.cleaned_text:
                print(f"Successfully extracted content from {url}")
                return article.cleaned_text
            else:
                print(f"No content extracted from {url}")
                return None

        except Exception as e:
            attempt += 1
            error_message = str(e)
            print(f"Attempt {attempt} failed for {url}: {error_message}")

            if attempt > retries:
                # Locking to avoid race conditions when updating the global error map
                with error_code_map_lock:
                    # Check if the error message is already mapped to an error code
                    if error_message in error_code_map:
                        error_code = error_code_map[error_message]
                    else:
                        # If it's a new error, assign a new error code
                        error_code = error_code_counter
                        error_code_map[error_message] = error_code
                        error_code_counter += 1
                        log_error_to_csv(url, error_code, error_message)
                        print(f"New error logged with code {error_code}")

                # Return the error code as the full_text for this essay
                return f"Error Code: {error_code}"

            if attempt <= retries:
                time.sleep(delay)  # Wait before retrying
                delay *= backoff_factor  # Increase the delay for the next retry

# Function to get URLs from Supabase
def get_urls_from_supabase(supabase, limit=100):
    try:
        # Get the total count
        count = supabase.table('urls_table') \
            .select('id', count='exact') \
            .or_('full_text.is.null,full_text.eq.') \
            .execute()
        
        total_count = count.count
        print(f"Total rows to process: {total_count}")

        # Fetch the limited number of URLs
        data = supabase.table('urls_table') \
            .select('id', 'url') \
            .or_('full_text.is.null,full_text.eq.') \
            .limit(limit) \
            .execute()

        return total_count, data.data  # Return both total count and the data

    except Exception as e:
        print(f"Failed to fetch URLs: {e}")
        return 0, []

# Function to update the full_text column in Supabase
def update_full_text_in_supabase(supabase, record_id, full_text):
    try:
        # Update the 'full_text' column for the corresponding record
        supabase.table('urls_table').update({'full_text': full_text}).eq('id', record_id).execute()
    except Exception as e:
        print(f"Failed to update full_text for ID {record_id}: {e}")

class ProxyNetworkFetcher(NetworkFetcher):
    def __init__(self, config, proxy):
        self.proxy = proxy
        super(ProxyNetworkFetcher, self).__init__(config)

    def fetch(self, url):
        sp_username = os.getenv('SP_USERNAME')
        sp_password = os.getenv('SP_PASSWORD')
        proxy = f"http://{sp_username}:{sp_password}@gate.smartproxy.com:7000"
        proxies = {'http': proxy, 'https': proxy}
        
        try:
            response = requests.get(url, proxies=proxies, timeout=10)
            response.raise_for_status()
            return response.text
        except (RequestException, SSLError) as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

# Function to log errors to a CSV file
def log_error_to_csv(url, error_code, error_message):
    with open(error_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([error_code, url, error_message])

# Function to log successful processing of URLs to a CSV file
def log_success_to_csv(url, record_id):
    with open(success_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([record_id, url, "Success"])

# Function to initialize the log CSVs with headers if they don't exist
def initialize_logs():
    if not os.path.exists(error_log_file):
        with open(error_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Error Code', 'URL', 'Error Message'])

    if not os.path.exists(success_log_file):
        with open(success_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Record ID', 'URL', 'Status'])

# Function to process a single URL
def process_url(record, supabase, index):
    if isinstance(record, dict) and 'id' in record and 'url' in record:
        record_id = record['id']
        url = record['url']
        print(f"Processing URL {index + 1}: {url}")  # Print the index and the URL

        # Scrape content
        full_text = scrape_content(url)

        if full_text:
            # Update the full_text column in Supabase
            update_full_text_in_supabase(supabase, record_id, full_text)
            log_success_to_csv(url, record_id)  # Log success
            print(f"Successfully updated full_text for record ID {record_id}")
        else:
            print(f"Skipping update for record ID {record_id}")
    else:
        print(f"Skipping invalid record: {record}")

# Main function to scrape essays and update Supabase using multithreading
def main():
    initialize_logs()  # Initialize the error and success log files
    supabase = connect_to_supabase()  # Initialize Supabase with provided URL and API key
    total_count, urls_records = get_urls_from_supabase(supabase, 100)

    print(f"Found {total_count} total URLs to process.")
    print(f"Processing {len(urls_records)} URLs in this batch.")

    # Use ThreadPoolExecutor to parallelize the scraping process
    with ThreadPoolExecutor(max_workers=50) as executor:  # Adjust max_workers based on your need
        futures = []
        
        # Submit tasks to the executor
        for index, record in enumerate(urls_records):
            futures.append(executor.submit(process_url, record, supabase, index))
        
        # Process results as they are completed
        for future in as_completed(futures):
            future.result()  # This will raise any exceptions caught in threads

if __name__ == "__main__":
    main()
