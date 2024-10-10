import requests
from bs4 import BeautifulSoup

def extract_essay_content(url):
    # Step 1: Get the HTML content of the page
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Failed to retrieve content. Status code: {response.status_code}"
    
    # Step 2: Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Step 3: Try to identify the main content container
    # You can adjust these selectors based on common HTML containers for articles
    possible_containers = [
        {'tag': 'article'},  # Many articles use <article> tag
        {'tag': 'div', 'class': 'content'},  # Content div
        {'tag': 'section', 'class': 'post-content'},  # Example section class
        {'tag': 'div', 'id': 'main-article'},  # Example id for main content
    ]
    
    for container in possible_containers:
        if 'class' in container:
            content = soup.find(container['tag'], class_=container['class'])
        elif 'id' in container:
            content = soup.find(container['tag'], id=container['id'])
        else:
            content = soup.find(container['tag'])
        
        if content:
            return content.get_text(separator=' ', strip=True)

    return "Content container not found."

# Example usage
url = 'https://www.statecraft.pub/p/how-to-stop-losing-17500-kidneys'
essay_content = extract_essay_content(url)

print("Extracted Essay Content:\n", essay_content)
