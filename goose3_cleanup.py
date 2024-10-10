from goose3 import Goose

def extract_article_content(url):
    g = Goose()
    article = g.extract(url=url)
    return article.cleaned_text

# Example usage
url = 'https://www.statecraft.pub/p/how-to-stop-losing-17500-kidneys'
content = extract_article_content(url)
print(content)