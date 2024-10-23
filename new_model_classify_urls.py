import torch
from transformers import BertTokenizer, BertForSequenceClassification
from supabase import create_client
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class URLTextClassifier:
    def __init__(self, model_path: str):
        """
        Initialize the classifier with a trained model
        model_path: Path to the saved model (e.g., 'model_outputs/best_model_20241022_181613')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Genre mapping (customize based on your genres)
        self.genre_mapping = {
            0: "narrative research",
            1: "commentary / op-ed",
            2: "journalism",
            3: "creative non-fiction",
            4: "articles/editorials"
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the text"""
        if not isinstance(text, str) or not text.strip():
            logging.warning(f"Invalid or empty text received: {text}")
            return ""
        text = text.strip()
        # Add any additional cleaning steps if needed
        return text

    def classify_text(self, text: str) -> Dict:
        """Classify a single text"""
        text = self.preprocess_text(text)
        if not text:
            logging.warning("Text is empty after preprocessing, skipping classification.")
            return {'genre_id': None, 'confidence': 0.0, 'genre_name': None}

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get prediction
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Map predicted class to genre name
            genre_name = self.genre_mapping.get(predicted_class, None)

            if genre_name is None:
                logging.warning(f"Predicted class {predicted_class} not found in genre mapping.")

            return {
                'genre_id': predicted_class,
                'confidence': confidence,
                'genre_name': genre_name
            }

        except Exception as e:
            logging.error(f"Error during classification: {e}")
            return {'genre_id': None, 'confidence': 0.0, 'genre_name': None}

def fetch_texts_from_supabase(limit: int, offset: int = 0, genre_empty_filter: bool = True) -> List[Dict]:
    """Fetch texts from Supabase URLs table in batches"""
    try:
        load_dotenv()
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )

        # Fetch rows where genre is NULL (or filter as necessary)
        query = supabase.table('urls_table').select('id, full_text').range(offset, offset + limit - 1)
        
        if genre_empty_filter:
            query = query.is_('genre', None)

        response = query.execute()

        if not response.data:
            logging.warning("No unclassified texts found in the database")
            return []

        logging.info(f"Successfully fetched {len(response.data)} texts (offset: {offset})")
        return response.data

    except Exception as e:
        logging.error(f"Error fetching texts from Supabase: {str(e)}")
        return []

def update_genre_in_supabase(supabase, text_id: int, genre_id: int, confidence: float) -> bool:
    """Update the genre in Supabase based on classification"""
    try:
        # Update the genre column (which now references the genres table)
        supabase.table('urls_table')\
            .update({
                'genre': genre_id,
                'classification_confidence': confidence
            })\
            .eq('id', text_id)\
            .execute()
        return True
    except Exception as e:
        logging.error(f"Error updating genre for text {text_id}: {str(e)}")
        return False

def classify_and_update(classifier: URLTextClassifier, text_data: Dict, supabase) -> Dict:
    """Classify and update a single text entry in Supabase"""
    text_id = text_data['id']
    full_text = text_data['full_text']

    # Classify text
    classification = classifier.classify_text(full_text)

    # Log if the classification result is missing or invalid
    if classification['genre_id'] is None or classification['genre_name'] is None:
        logging.warning(f"Failed to classify text with ID {text_id}. Classification result: {classification}")

    # Update Supabase
    success = update_genre_in_supabase(
        supabase,
        text_id,
        classification['genre_id'],
        classification['confidence']
    )

    # Return result
    return {
        'id': text_id,
        'genre_id': classification['genre_id'],
        'genre_name': classification['genre_name'],
        'confidence': classification['confidence'],
        'update_success': success
    }

def main():
    # Initialize Supabase client
    load_dotenv()
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_KEY')
    )

    # Initialize classifier
    model_path = 'fine_tuned_bert'  # Update with your model path
    classifier = URLTextClassifier(model_path)

    # Set parameters
    batch_size = 1000  # Maximum batch size per Supabase query
    workers = 50  # Number of threads/workers

    # Fetch texts in batches and process them with multi-threading
    total_processed = 0
    offset = 0
    results = []

    # Fetch and process until there are no more unclassified texts
    while True:
        # Fetch batch of texts
        texts = fetch_texts_from_supabase(limit=batch_size, offset=offset, genre_empty_filter=True)
        if not texts:
            break  # Stop if no more unclassified texts are found

        # Use ThreadPoolExecutor for multi-threading
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_text = {
                executor.submit(classify_and_update, classifier, text_data, supabase): text_data
                for text_data in texts
            }

            for future in tqdm(as_completed(future_to_text), total=len(future_to_text), desc="Classifying and updating texts"):
                result = future.result()
                results.append(result)
        
        total_processed += len(texts)
        offset += len(texts)  # Move to the next batch

    # Print results
    df = pd.DataFrame(results)
    print("\nClassification Results:")
    print(df.to_string(index=False))
    
    # Save results to CSV
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'classification_results_{timestamp}.csv', index=False)
    logging.info(f"Results saved to classification_results_{timestamp}.csv")

if __name__ == "__main__":
    main()
