import os
from openai import OpenAI
from supabase import create_client, Client
import re
from tqdm import tqdm

# Initialize Supabase client
url = "https://hyxoojvfuuvjcukjohyi.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase: Client = create_client(url, key)

# Initialize OpenAI client
client = OpenAI(api_key="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")

def keyword_check(text):
    keywords = ['roundup', 'digest', 'link list', 'interesting things', 'weekly links', 'monthly recap',
                'curated selection', 'top stories', 'reading list', 'what i\'m reading', 'link round-up']
    return any(keyword in text.lower() for keyword in keywords)

def identify_link_lists(start=0, batch_size=1000):
    total_processed = 0
    link_list_count = 0
    api_identified = 0
    keyword_identified = 0

    while True:
        # Fetch rows from the urls_table with pagination
        response = supabase.table("urls_table").select("id,title,description,genre").range(start, start + batch_size - 1).execute()
        rows = response.data

        if not rows:
            break  # No more rows to process

        print(f"Processing essays {start} to {start + len(rows) - 1}")

        for row in tqdm(rows, desc="Processing essays"):
            total_processed += 1
            title = row.get('title', '')
            description = row.get('description', '')
            combined_text = f"Title: {title}\nDescription: {description}"

            # Use OpenAI to analyze the text
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant that identifies if an essay is a link list, roundup, or digest of multiple articles or topics."},
                        {"role": "user", "content": f"Determine if the following text describes a link list, digest, or roundup of multiple articles or topics. Look for phrases like 'interesting things', 'digest', 'roundup', 'weekly links', 'monthly recap', or any indication that this is a collection of links or summaries. Respond with 'Yes' if it's likely a link list, or 'No' if it's not. Here's the text:\n\n{combined_text}"}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                is_link_list = response.choices[0].message.content.strip().lower() == 'yes'
                if is_link_list:
                    api_identified += 1
            except Exception as e:
                print(f"Error with OpenAI API for essay {row['id']}: {str(e)}")
                is_link_list = False

            # Fallback to keyword check if API doesn't identify as link list
            if not is_link_list:
                is_link_list = keyword_check(combined_text)
                if is_link_list:
                    keyword_identified += 1

            if is_link_list:
                # Safely update the genre column
                current_genre = row.get('genre', [])
                if current_genre is None:
                    current_genre = []
                elif isinstance(current_genre, str):
                    current_genre = [current_genre]
                
                if 'link list' not in current_genre:
                    new_genre = current_genre + ['link list']
                    try:
                        supabase.table("urls_table").update({"genre": new_genre}).eq("id", row['id']).execute()
                        print(f"\nUpdated genre for essay ID {row['id']}: Added 'link list'")
                        print(f"Title: {title}")
                        link_list_count += 1
                    except Exception as e:
                        print(f"\nError updating database for essay {row['id']}: {str(e)}")
                else:
                    print(f"\nEssay ID {row['id']} already has 'link list' genre. No changes made.")

        start += batch_size  # Move to the next batch

    print(f"\nTotal essays processed: {total_processed}")
    print(f"Total link lists identified: {link_list_count}")
    print(f"Identified by API: {api_identified}")
    print(f"Identified by keyword fallback: {keyword_identified}")

    return start  # Return the next starting point

if __name__ == "__main__":
    last_processed = 0
    while True:
        last_processed = identify_link_lists(start=last_processed)
        user_input = input("Continue to next batch? (y/n): ")
        if user_input.lower() != 'y':
            break
    
    print("Script completed.")