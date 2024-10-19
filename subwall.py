import requests
from bs4 import BeautifulSoup
import os
from supabase import create_client, Client
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

# Initialize Supabase client
url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase: Client = create_client(url, key)

def check_paywall_selenium(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    try:
        driver.get(url)
        time.sleep(3)  # Wait for the page to load, increase if necessary

        # Check for Substack-specific paywall elements
        try:
            paywall_element = driver.find_element(By.CLASS_NAME, "paywall")
            if paywall_element:
                print(f"Paywall detected for URL: {url}")
                return True
        except:
            pass

        # Check for subscription prompts
        if "Subscribe" in driver.page_source or "free trial" in driver.page_source:
            print(f"Paywall detected for URL: {url}")
            return True
    except Exception as e:
        print(f"Error accessing {url} with Selenium: {e}")
    finally:
        driver.quit()
    
    print(f"No paywall detected for URL: {url}")
    return False

def update_paywall_status(urls):
    for url in urls:
        is_paywalled = check_paywall_selenium(url)
        try:
            supabase.table("urls_table").update({"lock": is_paywalled}).eq("url", url).execute()
            print(f"Updated paywall status for URL: {url} to {is_paywalled}")
        except Exception as e:
            print(f"Error updating database for URL {url}: {e}")

def main():
    try:
        response = supabase.table("urls_table").select("url").execute()
        urls = [row["url"] for row in response.data]
        print(f"Retrieved {len(urls)} URLs from the database.")
    except Exception as e:
        print(f"Error retrieving URLs: {e}")
        return

    update_paywall_status(urls)

if __name__ == "__main__":
    main()
