import torch
from transformers import BertModel, BertTokenizer
from supabase import create_client, Client
import numpy as np
from tqdm import tqdm

# Supabase setup
SUPABASE_URL = "https://hyxoojvfuuvjcukjohyi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load your fine-tuned model
model = BertModel.from_pretrained('./fine_tuned_bert')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

def generate_embedding(text):
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        max_length=512, 
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Ensure we're using float64 (double precision)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float64)

batch_size = 1000
offset = 0
total_records = 0

while True:
    # Fetch a batch of data with pagination
    response = supabase.table("urls_table")\
    .select("id", "full_text")\
    .or_('content_embedding.is.null,content_embedding.eq.{}')\
    .limit(batch_size)\
    .execute()
    data = response.data

    if not data:
        # No more records to process
        break

    # Generate embeddings and update Supabase
    for item in tqdm(data, desc=f"Processing records {offset} to {offset + batch_size - 1}"):
        essay_id = item['id']
        full_text = item['full_text']
        content_embedding = item.get('content_embedding')

        if content_embedding is not None:
            print(f"Skipping essay {essay_id} because 'content_embedding' is already populated.")
            continue

        if not full_text:
            print(f"Skipping essay {essay_id} because 'full_text' is empty or None.")
            continue
        
        try:
            # Generate embedding
            embedding = generate_embedding(full_text)
            
            # Validate embedding
            if not np.all(np.isfinite(embedding)):
                print(f"Invalid embedding for essay {essay_id}. Skipping update.")
                continue

            # Convert numpy array to list of float64
            embedding_list = embedding.tolist()
            
            # Update Supabase
            response = supabase.table("urls_table").update({
                "content_embedding": embedding_list
            }).eq("id", essay_id).execute()
            
            print(f"Updated embedding for essay {essay_id}")
        except Exception as e:
            print(f"Error updating embedding for essay {essay_id}: {str(e)}")
    
    # Update the offset for the next batch
    offset += batch_size
    total_records += len(data)
    print(f"Processed {total_records} records so far.")

print("All embeddings generated and saved to Supabase.")