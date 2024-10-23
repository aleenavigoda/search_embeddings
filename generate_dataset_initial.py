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
from concurrent.futures import ThreadPoolExecutor
import time
from tqdm import tqdm
import psutil
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
import logging
import traceback
from functools import wraps  # Add this import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)

def error_handler(func):
    """Decorator for error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())
            raise
    return wrapper

@dataclass
class BalancedGenerationMetrics:
    target_size_bytes: int = 10 * 1024 * 1024  # 10MB
    current_size_bytes: int = 0
    min_samples_per_genre: int = 1000
    samples_per_genre: Dict[int, int] = None
    total_samples: int = 0
    failed_generations: int = 0
    
    def __post_init__(self):
        self.samples_per_genre = {i: 0 for i in range(5)}  # Genres 0-4
    
    def is_balanced(self) -> bool:
        return all(count >= self.min_samples_per_genre 
                  for count in self.samples_per_genre.values())
    
    def get_underrepresented_genres(self) -> List[int]:
        return [genre for genre, count in self.samples_per_genre.items()
                if count < self.min_samples_per_genre]

class RobustTextProcessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        self.stopwords = set(stopwords.words('english'))
        self.compile_patterns()
    
    def compile_patterns(self):
        self.cleanup_patterns = {
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s.,!?;:-]'),
            'email': re.compile(r'\S*@\S*\s?'),
            'url': re.compile(r'http\S+|www.\S+'),
            'multiple_periods': re.compile(r'\.{2,}'),
            'multiple_punctuation': re.compile(r'[!?]{2,}')
        }
        self.sentence_end = re.compile(r'[.!?]+\s+')

    def robust_sentence_split(self, text: str) -> List[str]:
        if not text:
            return []
        
        text = self.cleanup_patterns['multiple_periods'].sub('.', text)
        text = self.cleanup_patterns['multiple_punctuation'].sub('!', text)
        
        raw_sentences = self.sentence_end.split(text)
        valid_sentences = []
        
        for sentence in raw_sentences:
            cleaned = sentence.strip()
            if self.is_valid_sentence(cleaned):
                valid_sentences.append(cleaned)
        
        return valid_sentences

    def is_valid_sentence(self, text: str) -> bool:
        if not text:
            return False
        
        words = text.split()
        return (len(words) >= 3 and
                len(text) >= 10 and
                len(text) <= 1000 and
                any(c.isalpha() for c in text))

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        text = text.strip()
        for pattern in self.cleanup_patterns.values():
            text = pattern.sub(' ', text)
        
        words = text.split()
        if len(words) < 5 or len(words) > 1000:
            return ""
        
        return ' '.join(words)

class BalancedDataGenerator:
    def __init__(self):
        self.max_workers = 4
        self.batch_size = 50
        self.max_memory_gb = 1.0
        self.processor = RobustTextProcessor()
        self.data_dir = Path('generated_data')
        self.data_dir.mkdir(exist_ok=True)
        self.metrics = BalancedGenerationMetrics()
        self.variations_per_text = 5
        self.variation_methods = ['shuffle', 'subset', 'reverse', 'bookend']
    
    def monitor_resources(self) -> Dict:
        process = psutil.Process()
        return {
            'memory_gb': process.memory_info().rss / (1024 * 1024 * 1024),
            'cpu_percent': process.cpu_percent()
        }

    def generate_variation(self, sentences: List[str], method: str) -> str:
        if len(sentences) < 2:
            return ' '.join(sentences)
        
        if method == 'shuffle':
            np.random.shuffle(sentences)
            return ' '.join(sentences)
        elif method == 'subset':
            subset_size = max(len(sentences) // 2, 2)
            selected = np.random.choice(sentences, subset_size, replace=False)
            return ' '.join(selected)
        elif method == 'reverse':
            return ' '.join(sentences[::-1])
        elif method == 'bookend':
            if len(sentences) > 3:
                middle = sentences[1:-1]
                np.random.shuffle(middle)
                return ' '.join([sentences[0]] + middle + [sentences[-1]])
        
        return ' '.join(sentences)

    def generate_variations(self, text: str, genre_id: int) -> List[dict]:
        variations = []
        clean_text = self.processor.clean_text(text)
        
        if not clean_text:
            return variations
        
        sentences = self.processor.robust_sentence_split(clean_text)
        if len(sentences) < 2:
            return variations
        
        for _ in range(self.variations_per_text):
            method = np.random.choice(self.variation_methods)
            try:
                variation = self.generate_variation(sentences.copy(), method)
                variations.append({
                    'full_text': variation,
                    'genre_id': genre_id,
                    'text_length': len(variation),
                    'word_count': len(variation.split()),
                    'is_generated': True,
                    'generation_method': method,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logging.warning(f"Variation generation failed: {str(e)}")
                self.metrics.failed_generations += 1
        
        return variations

    def process_batch(self, texts: List[dict]) -> List[dict]:
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
                    self.metrics.failed_generations += 1
        
        return processed_texts

    @error_handler
    def generate_dataset(self, original_data: List[dict]) -> Tuple[pd.DataFrame, Dict]:
        start_time = time.time()
        all_generated_texts = []
        
        # Group by genre
        genre_groups = {}
        for item in original_data:
            genre_id = item.get('genre_id')
            if genre_id is not None and 0 <= genre_id <= 4:
                if genre_id not in genre_groups:
                    genre_groups[genre_id] = []
                genre_groups[genre_id].append(item)
        
        # Print original distribution
        logging.info("\nOriginal data distribution:")
        for genre_id in range(5):
            num_texts = len(genre_groups.get(genre_id, []))
            logging.info(f"Genre {genre_id}: {num_texts} texts")
        
        with tqdm(total=self.metrics.min_samples_per_genre * 5, 
                 desc="Generating balanced data", 
                 unit='samples') as pbar:
            
            while not self.metrics.is_balanced():
                underrepresented_genres = self.metrics.get_underrepresented_genres()
                
                for genre_id in underrepresented_genres:
                    if genre_id not in genre_groups or not genre_groups[genre_id]:
                        logging.error(f"No original texts for genre {genre_id}!")
                        continue
                    
                    num_needed = self.metrics.min_samples_per_genre - self.metrics.samples_per_genre[genre_id]
                    batch_size = min(self.batch_size, num_needed)
                    
                    batch = np.random.choice(
                        genre_groups[genre_id],
                        size=batch_size,
                        replace=True
                    )
                    
                    generated_texts = self.process_batch(batch)
                    
                    self.metrics.samples_per_genre[genre_id] += len(generated_texts)
                    self.metrics.total_samples += len(generated_texts)
                    
                    pbar.update(len(generated_texts))
                    all_generated_texts.extend(generated_texts)
                    
                    pbar.set_postfix({
                        'Genre': genre_id,
                        'Samples': self.metrics.samples_per_genre[genre_id],
                        'Failed': self.metrics.failed_generations
                    })
                    
                    if self.metrics.samples_per_genre[genre_id] >= self.metrics.min_samples_per_genre:
                        logging.info(f"Completed generation for genre {genre_id}")
        
        report = {
            'total_samples': self.metrics.total_samples,
            'samples_per_genre': self.metrics.samples_per_genre,
            'generation_time': time.time() - start_time,
            'failed_generations': self.metrics.failed_generations,
            'balance_metrics': {
                'min_samples': min(self.metrics.samples_per_genre.values()),
                'max_samples': max(self.metrics.samples_per_genre.values()),
                'balance_ratio': min(self.metrics.samples_per_genre.values()) / 
                                max(self.metrics.samples_per_genre.values())
            }
        }
        
        return pd.DataFrame(all_generated_texts), report

def main():
    load_dotenv()
    
    try:
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        
        logging.info("Fetching data from Supabase...")
        response = supabase.table("genre_assignments").select("*").execute()
        
        if not response.data:
            logging.error("No data found in Supabase!")
            return
        
        logging.info(f"Found {len(response.data)} original texts")
        
        # Initial genre distribution
        genre_counts = {}
        for item in response.data:
            genre_id = item.get('genre_id')
            if genre_id is not None and 0 <= genre_id <= 4:
                genre_counts[genre_id] = genre_counts.get(genre_id, 0) + 1
        
        logging.info("\nInitial genre distribution:")
        for genre_id in range(5):
            count = genre_counts.get(genre_id, 0)
            logging.info(f"Genre {genre_id}: {count} texts")
        
        generator = BalancedDataGenerator()
        df, report = generator.generate_dataset(response.data)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = generator.data_dir / f'balanced_dataset_{timestamp}.csv'
        report_path = generator.data_dir / f'generation_report_{timestamp}.json'
        
        df.to_csv(output_path, index=False)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logging.info("\nGeneration Complete!")
        logging.info(f"Total samples: {report['total_samples']:,}")
        logging.info("\nSamples per genre:")
        for genre_id in range(5):
            count = report['samples_per_genre'].get(str(genre_id), 0)
            logging.info(f"Genre {genre_id}: {count:,}")
        logging.info(f"\nBalance ratio: {report['balance_metrics']['balance_ratio']:.2f}")
        logging.info(f"Generation time: {report['generation_time']:.2f} seconds")
        logging.info(f"Failed generations: {report['failed_generations']}")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()