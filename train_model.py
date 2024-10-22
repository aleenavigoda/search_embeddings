import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
import json
from pathlib import Path
from tqdm import tqdm
import wandb  # for experiment tracking
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class GenreDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

class BERTGenreClassifier:
    def __init__(self, 
                 num_labels=5,
                 model_name='bert-base-uncased',
                 batch_size=16,
                 max_length=512,
                 learning_rate=2e-5,
                 num_epochs=5,
                 warmup_steps=0,
                 device=None):
        
        self.num_labels = num_labels
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Create output directory
        self.output_dir = Path('model_outputs')
        self.output_dir.mkdir(exist_ok=True)

    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for training"""
        texts = df['full_text'].tolist()
        labels = df['genre_id'].tolist()
        
        # Create dataset
        dataset = GenreDataset(texts, labels, self.tokenizer, self.max_length)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        return train_loader, val_loader

    def train(self, train_loader, val_loader):
        """Train the model"""
        # Initialize wandb
        wandb.init(project="genre-classification", config={
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.num_epochs,
        })
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}') as pbar:
                for batch in pbar:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    train_loss += loss.item()
                    train_steps += 1
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation phase
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics['accuracy'],
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f'best_model')
            
            # Print progress
            logging.info(f'\nEpoch {epoch+1}:')
            logging.info(f'Average training loss: {avg_train_loss:.4f}')
            logging.info(f'Validation loss: {val_loss:.4f}')
            logging.info(f'Validation metrics:')
            logging.info(json.dumps(val_metrics, indent=2))
        
        wandb.finish()

    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = val_loss / len(val_loader)
        metrics = classification_report(
            true_labels,
            predictions,
            output_dict=True
        )
        
        return val_loss, metrics

    def save_model(self, model_name):
        """Save the model and tokenizer"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f'{model_name}_{timestamp}'
        output_path.mkdir(exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logging.info(f'Model saved to {output_path}')

def main():
    # Load your generated dataset
    df = pd.read_csv('generated_data/balanced_dataset.csv')  # Update path as needed
    
    # Initialize classifier
    classifier = BERTGenreClassifier(
        num_labels=5,  # 0-4 genres
        batch_size=16,
        max_length=512,
        learning_rate=2e-5,
        num_epochs=5,
        warmup_steps=0
    )
    
    # Prepare data
    train_loader, val_loader = classifier.prepare_data(df)
    
    # Train model
    classifier.train(train_loader, val_loader)

if __name__ == "__main__":
    main()