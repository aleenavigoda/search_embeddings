import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm  # Progress bar
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bert_finetuning.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

# Fetch original data from Supabase
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

logging.info("Fetching original data from Supabase...")

supabase: Client = create_client(url, key)
response = supabase.table("genre_assignments").select("full_text", "genre_id").execute()
original_data = response.data

if not original_data:
    logging.error("No original data found in Supabase.")
else:
    logging.info(f"Loaded {len(original_data)} original data points from Supabase.")

# Load generated data from CSV
generated_data_path = 'generated_data/generated_dataset_20241023_131209.csv'  # Update to the actual path
logging.info(f"Loading generated data from CSV: {generated_data_path}")

try:
    generated_data = pd.read_csv(generated_data_path)
    logging.info(f"Loaded {len(generated_data)} generated data points from CSV.")
except FileNotFoundError as e:
    logging.error(f"Error loading generated data: {str(e)}")
    exit(1)

# Combine original and generated data
original_texts = [item['full_text'] for item in original_data]
original_labels = [item['genre_id'] - 1 for item in original_data]  # Adjusting labels from 1-5 to 0-4

generated_texts = generated_data['full_text'].tolist()
generated_labels = generated_data['genre_id'].tolist()

texts = original_texts + generated_texts
labels = original_labels + generated_labels

# Confirm that all data points have been loaded correctly
logging.info(f"Total data points for training: {len(texts)}")
assert len(texts) == len(labels), "Mismatch between texts and labels."

# Load pre-trained BERT model and tokenizer
logging.info("Loading BERT model and tokenizer...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)  # 5 genres

# Tokenize and encode the texts
logging.info("Tokenizing the texts...")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Convert labels to tensor
labels = torch.tensor(labels)

# Split the data
logging.info("Splitting data into training and validation sets...")
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

# Training loop with progress bar and logs
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

num_epochs = 2  # Increase the number of epochs as needed
logging.info(f"Starting training for {num_epochs} epochs on {device}...")

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")  # Add a progress bar for training
    
    for step, batch in enumerate(progress_bar):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Update progress bar with loss
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log training loss every 10 steps
        if step % 10 == 0 and step != 0:
            logging.info(f"  Step {step}/{len(train_loader)} - Loss: {loss.item():.4f}")
    
    avg_train_loss = total_loss / len(train_loader)
    logging.info(f"Average Training Loss for Epoch {epoch+1}: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")  # Progress bar for validation
    with torch.no_grad():
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Validation Loss for Epoch {epoch+1}: {avg_val_loss:.4f}\n")

# Save the fine-tuned model
save_directory = './fine_tuned_bert'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

logging.info(f"Model saved to {save_directory}")
