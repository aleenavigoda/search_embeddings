from supabase import create_client
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Initialize Supabase client
url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase = create_client(url, key)

# Ahrefs API setup
AHREFS_API_TOKEN = os.environ.get("7plB8ZQ6K0-84zIPK0lUfQnbiZ423UOmUWtodXAF")
AHREFS_BACKLINK_API_URL = "https://api.ahrefs.com/v3/"  # Ahrefs API endpoint

def get_backlinks(target_url):
    # Ahrefs API request parameters
    params = {
        "target": target_url,
        "mode": "domain",  # You might change this to "exact" or "subdomains" based on your needs
        "from": "backlinks",  # Getting backlinks data
        "output": "json",
        "limit": 10000,  # You can adjust this limit
        "token": AHREFS_API_TOKEN
    }

    response = requests.get(AHREFS_BACKLINK_API_URL, params=params)
    
    # Parse the response
    if response.status_code == 200:
        return response.json().get("backlinks", [])
    else:
        print(f"Error fetching backlinks: {response.status_code}, {response.text}")
        return []

def extract_outbound_links(full_text):
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(full_text, "html.parser")
    
    # Find all anchor tags with href attributes
    links = soup.find_all('a', href=True)
    
    outbound_links = []
    
    for link in links:
        href = link['href']
        
        # Parse the href to get the domain
        parsed_url = urlparse(href)
        
        # Filter out any non-HTTP/HTTPS links and exclude internal links (like mailto:, javascript:, etc.)
        if parsed_url.scheme in ['http', 'https']:
            outbound_links.append(href)
    
    return outbound_links

def populate_citation_index():
    # Fetch all URLs from our database
    result = supabase.table('urls_table').select('id, url, full_text').execute()
    our_urls = {item['url']: item['id'] for item in result.data}
    
    citations = []
    
    for item in result.data:
        essay_id = item['id']
        essay_url = item['url']
        full_text = item['full_text']  # Now we're storing the full text directly
        
        # Extract outbound links
        outbound_links = extract_outbound_links(full_text)
        for link in outbound_links:
            citations.append({
                'essay_id': essay_id,
                'citing_url': essay_url,
                'citation_type': 'outbound_link',
                'context': '',  # You might want to implement a function to extract context
                'is_internal': link in our_urls
            })
        
        # Get backlinks from Ahrefs API
        backlinks = get_backlinks(essay_url)
        for backlink in backlinks:
            citations.append({
                'essay_id': essay_id,
                'citing_url': backlink['referring_page'],  # Adjust field names based on Ahrefs response
                'citation_type': 'backlink',
                'context': backlink.get('anchor', ''),  # Anchor text is provided by Ahrefs
                'is_internal': backlink['referring_page'] in our_urls
            })
    
    # Insert all citations into the citation_index table
    for i in range(0, len(citations), 1000):  # Insert in batches of 1000
        supabase.table('citation_index').insert(citations[i:i+1000]).execute()

    print("Citation index population complete.")

# Run the function
populate_citation_index()
