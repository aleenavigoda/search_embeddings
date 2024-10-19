import requests
from bs4 import BeautifulSoup
import os
from supabase import create_client, Client

# Supabase connection details
url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"

supabase: Client = create_client(url, key)

def check_paywall(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Check for common paywall markers
            if "paywall" in soup.text.lower() or "subscription" in soup.text.lower():
                print(f"Paywall detected for URL: {url}")
                return True
        elif response.status_code == 402:  # HTTP 402 means payment required
            print(f"Paywall detected for URL (402 Payment Required): {url}")
            return True
    except requests.RequestException as e:
        print(f"Error accessing {url}: {e}")
    print(f"No paywall detected for URL: {url}")
    return False

def update_paywall_status(urls):
    data_to_update = []
    
    for url in urls:
        is_paywalled = check_paywall(url)
        data_to_update.append({"url": url, "lock": is_paywalled})
    
    try:
        for data in data_to_update:
            supabase.table("urls_table").update({"lock": data["lock"]}).eq("url", data["url"]).execute()
            print(f"Updated paywall status for URL: {data['url']} to {data['lock']}")
    except Exception as e:
        print(f"Error updating database: {e}")

def main():
    try:
        response = supabase.table("urls_table").select("url").execute()
        urls = [row["url"] for row in response.data]
        print(f"Successfully updated full_text for record ID {id}")
    except Exception as e:
        print(f"Error retrieving URLs: {e}")
        return
        

    update_paywall_status(urls)

if __name__ == "__main__":
    main()
