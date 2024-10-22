import torch
from transformers import BertTokenizer, BertForSequenceClassification
from supabase import create_client
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm

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
        if not isinstance(text, str):
            return ""
        text = text.strip()
        # Add any additional cleaning steps if needed
        return text

    def classify_text(self, text: str) -> Dict:
        """Classify a single text"""
        text = self.preprocess_text(text)
        if not text:
            return {'genre_id': None, 'confidence': 0.0}

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
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'genre_id': predicted_class,
            'confidence': confidence,
            'genre_name': self.genre_mapping[predicted_class]
        }

def fetch_texts_from_supabase(limit: int = 10) -> List[Dict]:
    """Fetch texts from Supabase URLs table"""
    try:
        load_dotenv()
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )

        # Fetch rows where genre_id is NULL, limit to specified number
        response = supabase.table('urls_table')\
            .select('id, full_text')\
            .is_('genre_id', None)\
            .limit(limit)\
            .execute()

        if not response.data:
            logging.warning("No unclassified texts found in the database")
            return []

        logging.info(f"Successfully fetched {len(response.data)} texts")
        return response.data

    except Exception as e:
        logging.error(f"Error fetching texts from Supabase: {str(e)}")
        return []

def update_genre_in_supabase(supabase, text_id: int, genre_id: int, confidence: float) -> bool:
    """Update the genre_id in Supabase"""
    try:
        supabase.table('urls_table')\
            .update({
                'genre_id': genre_id,
                'classification_confidence': confidence
            })\
            .eq('id', text_id)\
            .execute()
        return True
    except Exception as e:
        logging.error(f"Error updating genre for text {text_id}: {str(e)}")
        return False

def main():
    # Initialize Supabase client
    load_dotenv()
    supabase = create_client(
        os.getenv('SUPABASE_URL'),
        os.getenv('SUPABASE_KEY')
    )

    # Initialize classifier
    model_path = 'model_outputs/best_model_20241022_181613'  # Update with your model path
    classifier = URLTextClassifier(model_path)

    # Fetch texts
    texts = fetch_texts_from_supabase(limit=10)
    if not texts:
        logging.error("No texts to classify")
        return

    # Process each text
    results = []
    for text_data in tqdm(texts, desc="Classifying texts"):
        text_id = text_data['id']
        full_text = text_data['full_text']

        # Classify text
        classification = classifier.classify_text(full_text)
        
        # Update Supabase
        if classification['genre_id'] is not None:
            success = update_genre_in_supabase(
                supabase,
                text_id,
                classification['genre_id'],
                classification['confidence']
            )
            
            # Store result
            results.append({
                'id': text_id,
                'genre_id': classification['genre_id'],
                'genre_name': classification['genre_name'],
                'confidence': classification['confidence'],
                'update_success': success
            })

    # Print results
    print("\nClassification Results:")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(f'classification_results_{timestamp}.csv', index=False)
    logging.info(f"Results saved to classification_results_{timestamp}.csv")

if __name__ == "__main__":
    main()