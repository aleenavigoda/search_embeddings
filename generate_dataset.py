import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from supabase import create_client
import json
from datetime import datetime
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import logging
from typing import List, Dict, Tuple, Optional
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Downloaded NLTK resources successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")

# Download NLTK resources at startup
download_nltk_resources()

class TextProcessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.compile_patterns()
    
    def compile_patterns(self):
        """Compile regex patterns for text processing"""
        self.cleanup_patterns = {
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s.,!?;:-]'),
            'email': re.compile(r'\S*@\S*\s?'),
            'url': re.compile(r'http\S+|www.\S+')
        }

class TextProcessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.compile_patterns()  # Make sure this is called
    
    def compile_patterns(self):
        """Compile regex patterns for text processing"""
        self.cleanup_patterns = {
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s.,!?;:-]'),
            'email': re.compile(r'\S*@\S*\s?'),
            'url': re.compile(r'http\S+|www.\S+')
        }

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules"""
        # First clean the text
        text = self.clean_text(text)
        if not text:
            return []
            
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def clean_text(self, text: str) -> str:
        """Clean text while maintaining essential structure"""
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        # Apply cleanup patterns
        for pattern in self.cleanup_patterns.values():
            text = pattern.sub(' ', text)
        
        return ' '.join(text.split())


class DataGenerator:
    def __init__(self):
        self.processor = TextProcessor()
        self.data_dir = Path('generated_data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Genre configurations
        self.genre_mapping = {
            0: "narrative research",
            1: "commentary / op-ed",
            2: "journalism",
            3: "creative non-fiction",
            4: "articles/editorials"
        }
        
        # Generation parameters
        self.variations_per_text = 3
        self.min_samples_per_genre = 1000
        self.max_workers = 4

    def generate_variation(self, text: str, method: str) -> str:
        """Generate a variation of the text"""
        sentences = self.processor.split_into_sentences(text)
        if len(sentences) < 2:
            return text
            
        if method == 'shuffle':
            # Shuffle while keeping some structure
            if len(sentences) > 2:
                middle = sentences[1:-1]
                random.shuffle(middle)
                return ' '.join([sentences[0]] + middle + [sentences[-1]])
            return text
            
        elif method == 'subset':
            # Take a subset of sentences while maintaining meaning
            if len(sentences) > 3:
                num_sentences = max(3, len(sentences) // 2)
                selected = [sentences[0]]  # Keep first sentence
                selected.extend(random.sample(sentences[1:-1], min(num_sentences-2, len(sentences)-2)))
                selected.append(sentences[-1])  # Keep last sentence
                return ' '.join(selected)
            return text
            
        return text

    def generate_variations(self, text: str, genre_id: int) -> List[dict]:
        """Generate multiple variations of a text"""
        variations = []
        methods = ['shuffle', 'subset']
        
        cleaned_text = self.processor.clean_text(text)
        if not cleaned_text:
            return variations

        for _ in range(self.variations_per_text):
            method = random.choice(methods)
            variation = self.generate_variation(cleaned_text, method)
            
            if variation and variation != cleaned_text:
                variations.append({
                    'full_text': variation,
                    'genre_id': genre_id,
                    'text_length': len(variation),
                    'word_count': len(variation.split()),
                    'is_generated': True,
                    'generation_method': method,
                    'timestamp': datetime.now().isoformat()
                })
        
        return variations

    def process_batch(self, texts: List[dict]) -> List[dict]:
        """Process a batch of texts in parallel"""
        processed_texts = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.generate_variations, text['full_text'], text['genre_id'])
                for text in texts
            ]
            
            for future in futures:
                try:
                    variations = future.result(timeout=10)
                    processed_texts.extend(variations)
                except Exception as e:
                    logging.warning(f"Batch processing error: {str(e)}")
        
        return processed_texts


    def generate_dataset(self, original_data: List[dict]) -> Tuple[pd.DataFrame, Dict]:
        """Generate balanced dataset"""
        start_time = time.time()
        all_generated_texts = []
        samples_per_genre = {i: 0 for i in range(5)}  # Track samples for each genre
        
        # First, include all original data
        logging.info("\nProcessing original texts...")
        for item in original_data:
            genre_id = item.get('genre_id')
            if genre_id is not None and 0 <= genre_id <= 4:
                all_generated_texts.append({
                    'full_text': item['full_text'],
                    'genre_id': genre_id,
                    'text_length': len(item['full_text']),
                    'word_count': len(item['full_text'].split()),
                    'is_generated': False,
                    'generation_method': 'original',
                    'timestamp': datetime.now().isoformat()
                })
                samples_per_genre[genre_id] += 1

        # Print original distribution
        logging.info("\nOriginal data distribution:")
        for genre_id in range(5):
            num_texts = samples_per_genre[genre_id]
            logging.info(f"Genre {genre_id} ({self.genre_mapping[genre_id]}): {num_texts} texts")

        # Generate additional data until we reach minimum samples for each genre
        with tqdm(total=self.min_samples_per_genre * 5, 
                 desc="Generating balanced data", 
                 unit='samples') as pbar:
            
            while min(samples_per_genre.values()) < self.min_samples_per_genre:
                for genre_id in range(5):
                    if samples_per_genre[genre_id] >= self.min_samples_per_genre:
                        continue
                        
                    # Get original texts for this genre
                    genre_texts = [
                        item for item in original_data 
                        if item.get('genre_id') == genre_id
                    ]
                    
                    if not genre_texts:
                        logging.error(f"No original texts for genre {genre_id}")
                        continue
                    
                    # Process texts in batches
                    batch_size = min(len(genre_texts), 5)  # Process 5 texts at a time
                    batch = random.sample(genre_texts, batch_size)
                    generated_texts = self.process_batch(batch)
                    
                    if generated_texts:
                        all_generated_texts.extend(generated_texts)
                        new_samples = len(generated_texts)
                        samples_per_genre[genre_id] += new_samples
                        pbar.update(new_samples)
                    
                    pbar.set_postfix({
                        'Genre': f"{genre_id} ({samples_per_genre[genre_id]})"
                    })
        
        # Generate report
        generation_time = time.time() - start_time
        report = {
            'total_samples': len(all_generated_texts),
            'samples_per_genre': samples_per_genre,
            'generation_time': generation_time,
            'genre_mapping': self.genre_mapping,
            'original_distribution': {
                genre_id: len([x for x in original_data if x.get('genre_id') == genre_id])
                for genre_id in range(5)
            }
        }
        
        return pd.DataFrame(all_generated_texts), report

def main():
    """Main execution function"""
    load_dotenv()
    
    try:
        # Initialize Supabase client
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        
        # Fetch data from Supabase
        logging.info("Fetching data from Supabase...")
        response = supabase.table("genre_assignments").select("*").execute()
        
        if not response.data:
            logging.error("No data found in Supabase!")
            return
        
        logging.info(f"Found {len(response.data)} original texts")
        
        # Initialize generator and generate dataset
        generator = DataGenerator()
        df, report = generator.generate_dataset(response.data)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save dataset
        output_path = generator.data_dir / f'generated_dataset_{timestamp}.csv'
        df.to_csv(output_path, index=False)
        
        # Save report
        report_path = generator.data_dir / f'generation_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Print summary
        logging.info("\nGeneration Complete!")
        logging.info(f"Total samples: {report['total_samples']:,}")
        logging.info("\nSamples per genre:")
        for genre_id in range(5):
            genre_name = generator.genre_mapping[genre_id]
            count = report['samples_per_genre'][genre_id]
            logging.info(f"Genre {genre_id} ({genre_name}): {count:,}")
        logging.info(f"\nGeneration time: {report['generation_time']:.2f} seconds")
        logging.info(f"Output saved to: {output_path}")
        logging.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()        