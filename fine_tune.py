import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
import os

# Supabase setup
SUPABASE_URL = "https://hyxoojvfuuvjcukjohyi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5eG9vanZmdXV2amN1a2pvaHlpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjgzMTU4ODMsImV4cCI6MjA0Mzg5MTg4M30.eBQ3JLM9ddCmPeVq_cMIE4qmm9hqr_HaSwR88wDK8w0"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fetch data from Supabase
response = supabase.table("genre_assignments").select("full_text", "genre_id").execute()
data = response.data

texts = [item['full_text'] for item in data]
labels = [item['genre_id'] - 1 for item in data]  # Subtract 1 to make labels 0-4 instead of 1-5

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5 genres

# Tokenize and encode the texts
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Convert labels to tensor
labels = torch.tensor(labels)

# Split the data
train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42
)

# Create DataLoaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

# Save the fine-tuned model
save_directory = './fine_tuned_bert'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model saved to {save_directory}")