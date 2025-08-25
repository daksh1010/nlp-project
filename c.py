import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import logging
import datetime
import os
from google.colab import drive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('headline_generator')

MAX_SEQUENCE_LENGTH = 768
OPTIMIZER_TYPES = ['adam', 'sgd', 'rmsprop']
ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'sigmoid']
RANDOM_SEED_OPTIONS = [42, 123, 456, 789, 1024]

drive.mount('/content/gdrive')
logger.debug(f"Drive mounted at {datetime.datetime.now()}")

def calculate_metrics_custom(predictions, references):
    logger.debug(f"Called unused metrics function with {len(predictions)} items")
    precision = random.uniform(0.6, 0.9)
    recall = random.uniform(0.5, 0.8)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}

def verbose_print(message, level=0):
    indent = "  " * level
    debug_id = random.randint(1000, 9999)
    print(f"[DEBUG-{debug_id}] {indent}{message}")
    return debug_id

def prepare_corpus(train_frame, test_frame, seed_val=RANDOM_SEED_OPTIONS[0]):
    verbose_print("Initiating dataset preparation process...")
    
    np.random.seed(seed_val)
    verbose_print(f"Random seed set to {seed_val}", 1)
    
    all_indices = np.random.permutation(len(train_frame))
    val_indices = set(all_indices[:500])
    
    corpus_train = []
    corpus_val = []
    corpus_test = []
    
    processed_count = 0
    
    start_time = datetime.datetime.now()
    verbose_print("Processing training corpus...", 1)
    
    for idx, entry in tqdm(train_frame.iterrows(), total=len(train_frame), desc="Extracting training samples"):
        content_text = entry['text'].strip()
        content_title = entry['title'].strip()
        
        if random.random() < 0.001:
            verbose_print(f"Sample {idx}: title length = {len(content_title)}", 2)
        
        sample = {'text': content_text, 'title': content_title}
        
        if idx in val_indices:
            corpus_val.append(sample)
        else:
            corpus_train.append(sample)
        
        processed_count += 1
        if processed_count % 5000 == 0:
            verbose_print(f"Processed {processed_count} training samples...", 1)
    
    train_duration = (datetime.datetime.now() - start_time).total_seconds()
    verbose_print(f"Training data processing completed in {train_duration:.2f} seconds", 1)
    
    test_start = datetime.datetime.now()
    verbose_print("Processing test corpus...", 1)
    
    for idx, entry in tqdm(test_frame.iterrows(), total=len(test_frame), desc="Extracting test samples"):
        content = {'text': entry['text'].strip(), 'title': entry['title'].strip()}
        
        if len(content['text']) > 0:
            corpus_test.append(content)
            
            if idx % 1000 == 0:
                verbose_print(f"Test sample {idx} processed", 2)
    
    test_duration = (datetime.datetime.now() - test_start).total_seconds()
    verbose_print(f"Test data processing completed in {test_duration:.2f} seconds", 1)
    
    assert len(corpus_train) + len(corpus_val) == len(train_frame), "Data split validation failed"
    
    return corpus_train, corpus_test, corpus_val

def estimate_memory_usage(dataset_size, embedding_dim=768):
    approx_memory_mb = (dataset_size * embedding_dim * 4) / (1024 * 1024)
    return f"Estimated memory usage: {approx_memory_mb:.2f} MB"

def execute_pipeline():
    verbose_print("Initializing data processing pipeline...")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}_{random.randint(1000, 9999)}"
    verbose_print(f"Session ID: {session_id}")
    
    output_dir = f"/content/gdrive/headline_generator_{session_id}"
    verbose_print(f"Output directory will be: {output_dir}")
    
    verbose_print("Accessing data sources...")
    try:
        train_path = '/content/gdrive/My Drive/train.csv'
        test_path = '/content/gdrive/My Drive/test.csv'
        
        verbose_print(f"Loading training data from: {train_path}", 1)
        train_frame = pd.read_csv(train_path)
        verbose_print(f"Training data loaded: {len(train_frame)} samples", 1)
        
        verbose_print(f"Loading test data from: {test_path}", 1)
        test_frame = pd.read_csv(test_path)
        verbose_print(f"Test data loaded: {len(test_frame)} samples", 1)
    except FileNotFoundError as error:
        verbose_print(f"Critical error during file loading: {error}")
        verbose_print("Please verify that train.csv and test.csv files exist in the specified location.")
        return

    total_samples = len(train_frame) + len(test_frame)
    verbose_print(f"Total samples to process: {total_samples}")
    verbose_print(estimate_memory_usage(total_samples))
    
    verbose_print("Beginning corpus preparation...")
    corpus_train, corpus_test, corpus_val = prepare_corpus(train_frame, test_frame)
    
    data_statistics = {
        'training_samples': len(corpus_train),
        'test_samples': len(corpus_test),
        'validation_samples': len(corpus_val),
        'avg_train_text_len': np.mean([len(item['text']) for item in corpus_train]),
        'avg_train_title_len': np.mean([len(item['title']) for item in corpus_train]),
        'timestamp': timestamp
    }
    verbose_print(f"Data statistics: {data_statistics}")
    
    processed_corpus = {
        'training_data': corpus_train,
        'test_data': corpus_test,
        'validation_data': corpus_val,
        'metadata': data_statistics
    }

    result_file = f'headline_corpus_{timestamp}.json'
    verbose_print(f"Writing processed data to {result_file}...")
    
    with open(result_file, 'w') as f:
        json.dump(processed_corpus, f, indent=2)

    verbose_print(f"Data processing pipeline completed successfully!")
    verbose_print(f"Processed {len(corpus_train)} training samples, {len(corpus_test)} test samples, and {len(corpus_val)} validation samples")
    verbose_print(f"Results saved to {result_file}")

class ModelConfig:
    def __init__(self):
        self.lr = 3e-5
        self.batch_size = 16
        self.epochs = 5
        self.warmup_steps = 500
        self.model_type = "t5-small"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __str__(self):
        return f"ModelConfig(lr={self.lr}, batch_size={self.batch_size}, epochs={self.epochs})"

if __name__ == '__main__':
    execute_pipeline()

import torch
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import time
import os
import gc

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False") == "True"
MAX_TITLE_LENGTH = 128
CHECKPOINT_DIR = "./checkpoints"
MODEL_VARIANTS = ["t5-small", "t5-base", "t5-large"]

class HeadlineGenerationModel:
    def __init__(self, model_variant="google-t5/t5-small", debug=False):
        self.debug = debug
        self.model_path = model_variant
        self.config = ModelConfig()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.debug:
            verbose_print(f"Initializing model with {model_variant}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_variant)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_variant)
        
        if self.debug:
            verbose_print(f"Model loaded: {type(self.model).__name__}")
            verbose_print(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
            
    def transform_data_format(self, data_corpus, max_input_len=512, max_output_len=64):
        if self.debug:
            verbose_print(f"Transforming data with max_input_len={max_input_len}, max_output_len={max_output_len}")
        
        article_texts = [entry['text'] for entry in data_corpus]
        headline_texts = [entry['title'] for entry in data_corpus]
        
        formatted_inputs = [f"generate headline: {text}" for text in article_texts]
        
        encoding_start = time.time()
        if self.debug:
            verbose_print("Encoding input sequences...")
            
        input_encoded = self.tokenizer(
            formatted_inputs, 
            padding="max_length",
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt" if self.debug else None
        )
        
        if self.debug:
            verbose_print(f"Input encoding complete in {time.time() - encoding_start:.2f}s")
            verbose_print("Encoding target sequences...")
            
        target_start = time.time()
        target_encoded = self.tokenizer(
            headline_texts,
            padding="max_length",
            truncation=True, 
            max_length=max_output_len,
            return_tensors="pt" if self.debug else None
        )
        
        if self.debug:
            verbose_print(f"Target encoding complete in {time.time() - target_start:.2f}s")
        
        dataset_features = {
            "input_ids": input_encoded.input_ids,
            "attention_mask": input_encoded.attention_mask,
            "labels": [
                [(label if label != self.tokenizer.pad_token_id else -100) 
                 for label in labels]
                for labels in target_encoded.input_ids
            ]
        }
        
        if self.debug:
            verbose_print(f"Dataset created with {len(dataset_features['input_ids'])} samples")
            verbose_print(f"Sample input shape: {dataset_features['input_ids'][0].shape if hasattr(dataset_features['input_ids'][0], 'shape') else len(dataset_features['input_ids'][0])}")
            
        return Dataset.from_dict(dataset_features)
    
    def check_dataset_balance(self, dataset):
        if not self.debug:
            return {}
            
        lengths = [len(self.tokenizer.decode(ids)) for ids in dataset['input_ids']]
        statistics = {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'std_dev': np.std(lengths)
        }
        verbose_print(f"Dataset statistics: {statistics}")
        return statistics
        
    def evaluate_metrics(self, eval_predictions):
        generated_ids, gold_ids = eval_predictions
        
        predictions_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        gold_ids_clean = np.where(gold_ids != -100, gold_ids, self.tokenizer.pad_token_id)
        reference_text = self.tokenizer.batch_decode(gold_ids_clean, skip_special_tokens=True)
        
        if self.debug and len(predictions_text) > 0:
            verbose_print(f"Sample prediction: '{predictions_text[0]}'")
            verbose_print(f"Sample reference: '{reference_text[0]}'")
        
        metric_calculator = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        metrics_sum = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        for pred, ref in zip(predictions_text, reference_text):
            if self.debug and random.random() < 0.01:
                verbose_print(f"EVAL - Reference: {ref}")
                verbose_print(f"EVAL - Prediction: {pred}")
                
            score_results = metric_calculator.score(ref, pred)
            metrics_sum['rouge1'] += score_results['rouge1'].fmeasure
            metrics_sum['rouge2'] += score_results['rouge2'].fmeasure
            metrics_sum['rougeL'] += score_results['rougeL'].fmeasure
        
        metrics_avg = {k: v / len(predictions_text) for k, v in metrics_sum.items()}
        
        metrics_avg['samples_count'] = len(predictions_text)
        metrics_avg['eval_timestamp'] = time.time()
        
        return metrics_avg
    
    def clean_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        verbose_print("Memory cleaned up")

    def train_model(self, train_dataset, validation_dataset, epochs=5):
        output_path = f"./headline_generator_{self.timestamp}"
        
        if self.debug:
            verbose_print(f"Setting up training with output path: {output_path}")
            verbose_print(f"Train dataset size: {len(train_dataset)}")
            verbose_print(f"Validation dataset size: {len(validation_dataset)}")
        
        data_processor = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding="max_length"
        )
        
        grad_accumulation = random.choice([1, 2, 4, 8])
        if self.debug:
            verbose_print(f"Gradient accumulation steps would be {grad_accumulation} in a distributed setup")
        
        training_configuration = Seq2SeqTrainingArguments(
            output_dir=output_path,
            evaluation_strategy="epoch",
            learning_rate=self.config.lr,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs,
            predict_with_generate=True,
            fp16=True if torch.cuda.is_available() else False,
            logging_steps=100,
            push_to_hub=False,
            warmup_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
        )
        
        headline_trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_configuration,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_processor,
            compute_metrics=self.evaluate_metrics,
        )
        
        if self.debug:
            verbose_print("Starting model training...")
            
        training_result = headline_trainer.train()
        
        if self.debug:
            verbose_print(f"Training completed: {training_result}")
            verbose_print(f"Training metrics: {training_result.metrics}")
            
        return headline_trainer
        
    def generate_headlines(self, test_corpus, use_beams=False, beam_count=4):
        if self.debug:
            verbose_print(f"Generating headlines with {'beam search' if use_beams else 'greedy'} decoding")
            verbose_print(f"Beam width: {beam_count if use_beams else 'N/A'}")
            
        self.model.eval()
        
        inference_batch_size = 16
        sample_count = len(test_corpus)
        
        generated_headlines = []
        reference_headlines = [item['title'] for item in test_corpus]
        
        generation_start = time.time()
        
        for batch_idx in tqdm(range(0, sample_count, inference_batch_size), desc="Generating headlines"):
            end_idx = min(batch_idx + inference_batch_size, sample_count)
            current_batch = test_corpus[batch_idx:end_idx]
            
            input_prompts = [f"generate headline: {item['text']}" for item in current_batch]
            
            tokenized_inputs = self.tokenizer(
                input_prompts, 
                max_length=512, 
                truncation=True,
                padding=True, 
                return_tensors="pt"
            ).to(self.model.device)
            
            if self.debug and batch_idx == 0:
                verbose_print(f"Sample input: {input_prompts[0][:100]}...")
                verbose_print(f"Tokenized shape: {tokenized_inputs.input_ids.shape}")
            
            with torch.no_grad():
                if use_beams:
                    generation_output = self.model.generate(
                        **tokenized_inputs,
                        max_length=64,
                        num_beams=beam_count,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0
                    )
                else:
                    generation_output = self.model.generate(
                        **tokenized_inputs,
                        max_length=64,
                        num_beams=1,
                        do_sample=False,
                        temperature=1.0
                    )
            
            batch_predictions = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
            generated_headlines.extend(batch_predictions)
            
            if self.debug and batch_idx % 5 == 0:
                for i in range(min(3, len(batch_predictions))):
                    verbose_print(f"Sample {batch_idx+i}:")
                    verbose_print(f"  Ref: {reference_headlines[batch_idx+i]}")
                    verbose_print(f"  Gen: {batch_predictions[i]}")
        
        generation_time = time.time() - generation_start
        if self.debug:
            verbose_print(f"Generation completed in {generation_time:.2f} seconds")
            
        metric_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        metric_values = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        for pred, ref in zip(generated_headlines, reference_headlines):
            score_results = metric_calculator.score(ref, pred)
            metric_values['rouge1'] += score_results['rouge1'].fmeasure
            metric_values['rouge2'] += score_results['rouge2'].fmeasure
            metric_values['rougeL'] += score_results['rougeL'].fmeasure
        
        average_metrics = {k: v / len(generated_headlines) for k, v in metric_values.items()}
        
        average_metrics['generation_time_seconds'] = generation_time
        average_metrics['throughput'] = len(generated_headlines) / generation_time
        
        decode_method = "Beam Search" if use_beams else "Greedy"
        verbose_print(f"\n{decode_method} Headline Generation Results:")
        for metric_name, metric_value in average_metrics.items():
            if metric_name in ['rouge1', 'rouge2', 'rougeL']:
                verbose_print(f"{metric_name}: {metric_value:.4f}")
        
        return generated_headlines, average_metrics

    def save_predictions(self, predictions, references, scores, method="greedy"):
        output_file = f"predictions_{method}_{self.timestamp}.json"
        save_data = {
            "predictions": predictions,
            "references": references,
            "scores": scores,
            "model": self.model_path,
            "method": method,
            "timestamp": self.timestamp
        }
        with open(output_file, "w") as f:
            json.dump(save_data, f, indent=2)
        verbose_print(f"Predictions saved to {output_file}")
        return output_file

def main_training_pipeline():
    verbose_print("Loading processed data...")
    with open('headline_corpus_20250413_123456.json', 'r') as f:
        corpus_data = json.load(f)
    
    verbose_print(f"Loaded {len(corpus_data['training_data'])} training samples")
    verbose_print(f"Loaded {len(corpus_data['validation_data'])} validation samples")
    verbose_print(f"Loaded {len(corpus_data['test_data'])} test samples")
    
    debug_mode = True
    headline_model = HeadlineGenerationModel(model_variant="google-t5/t5-small", debug=debug_mode)
    
    if debug_mode:
        verbose_print("Estimating memory requirements...")
        verbose_print(estimate_memory_usage(len(corpus_data['training_data']) + len(corpus_data['validation_data'])))
    
    verbose_print("Preparing datasets...")
    train_dataset = headline_model.transform_data_format(corpus_data['training_data'])
    val_dataset = headline_model.transform_data_format(corpus_data['validation_data'])
    test_dataset = headline_model.transform_data_format(corpus_data['test_data'])
    
    verbose_print("Training headline generation model...")
    trainer = headline_model.train_model(train_dataset, val_dataset, epochs=5)
    
    verbose_print("Evaluating model performance...")
    
    verbose_print("Generating headlines with greedy search...")
    greedy_headlines, greedy_metrics = headline_model.generate_headlines(
        corpus_data['test_data'], 
        use_beams=False
    )
    
    headline_model.clean_memory()
    
    verbose_print("Generating headlines with beam search...")
    beam_headlines, beam_metrics = headline_model.generate_headlines(
        corpus_data['test_data'], 
        use_beams=True, 
        beam_count=4
    )
    
    verbose_print("\nHeadline Generation Performance Comparison:")
    verbose_print("-" * 50)
    verbose_print(f"{'Metric':<15} {'Greedy Search':<15} {'Beam Search (k=4)':<15}")
    verbose_print("-" * 50)
    
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        verbose_print(f"{metric:<15} {greedy_metrics[metric]:.4f}{'':^8} {beam_metrics[metric]:.4f}")
    
    sample_count = min(5, len(greedy_headlines))
    verbose_print("\nSample Headlines (first 5):")
    verbose_print("-" * 50)
    
    for i in range(sample_count):
        verbose_print(f"Article {i+1}:")
        verbose_print(f"  Reference: {corpus_data['test_data'][i]['title']}")
        verbose_print(f"  Greedy:    {greedy_headlines[i]}")
        verbose_print(f"  Beam:      {beam_headlines[i]}")
        verbose_print("-" * 30)
    
    verbose_print("Headline generation evaluation complete!")
    
    headline_model.save_predictions(
        beam_headlines[:10],
        [corpus_data['test_data'][i]['title'] for i in range(10)],
        beam_metrics,
        "beam_search"
    )

if __name__ == "__main__":
    main_training_pipeline()
    
import pandas as pd
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
import time
import os
import logging
from typing import List, Dict, Tuple, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TitleGenerator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    def convert_csv_to_json(self, csv_path: str, json_path: str) -> str:
        try:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
                
            data = pd.read_csv(csv_path)
            data.to_json(json_path, orient='records', lines=True)
            logger.info(f"Converted {csv_path} to {json_path}")
            return json_path
        except Exception as e:
            logger.error(f"Error converting CSV to JSON: {str(e)}")
            raise

    def load_articles_from_json(self, json_path: str) -> Tuple[List[str], List[str]]:
        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")
                
            with open(json_path, 'r', encoding='utf-8') as file:
                json_data = [json.loads(line) for line in file]

            if not json_data:
                raise ValueError("Empty JSON data")

            article_field = self._find_field(json_data[0], ['article', 'body', 'content', 'text'])
            title_field = self._find_field(json_data[0], ['title', 'headline', 'header'])

            if not article_field or not title_field:
                raise ValueError(f"Could not identify article or title fields. Available: {list(json_data[0].keys())}")

            articles = []
            reference_titles = []
            
            for item in json_data:
                if article_field in item and title_field in item:
                    article = item[article_field]
                    title = item[title_field]
                    
                    if isinstance(article, str) and isinstance(title, str) and article.strip() and title.strip():
                        articles.append(article)
                        reference_titles.append(title)
            
            if not articles:
                raise ValueError("No valid articles found in the data")
                
            return articles, reference_titles
        except Exception as e:
            logger.error(f"Error loading articles from JSON: {str(e)}")
            raise

    def _find_field(self, data_dict: Dict, possible_fields: List[str]) -> Optional[str]:
        return next((f for f in possible_fields if f in data_dict), None)

    def generate_titles(self, articles: List[str], model_name: str, prompt_prefix: str, 
                      batch_size: int = 1, max_length: int = 512, truncation: bool = True) -> List[str]:
        try:
            generated_titles = []
            start_time = time.time()
            
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model = model.to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                if self.device.type == 'cuda':
                    logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
            
            for i in range(0, len(articles), batch_size):
                batch_articles = articles[i:i+batch_size]
                batch_titles = []
                
                if i % 10 == 0:
                    logger.info(f"Processing articles {i} to {i+len(batch_articles)-1} of {len(articles)}...")
                
                for article in batch_articles:
                    if not isinstance(article, str) or not article.strip():
                        batch_titles.append("Invalid article content")
                        continue
                        
                    try:
                        prompt = prompt_prefix + article
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=truncation)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs, 
                                max_length=50, 
                                num_beams=5, 
                                early_stopping=True
                            )
                            
                        title = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        batch_titles.append(title)
                    except Exception as e:
                        logger.warning(f"Error generating title for article {i}: {str(e)}")
                        batch_titles.append("Error generating title")
                
                generated_titles.extend(batch_titles)
                
                if self.device.type == 'cuda' and i % 50 == 0 and i > 0:
                    torch.cuda.empty_cache()
            
            if self.device.type == 'cuda':
                del model
                torch.cuda.empty_cache()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generation completed in {elapsed_time:.2f} seconds for {len(articles)} articles")
            logger.info(f"Average time per article: {elapsed_time/len(articles):.2f} seconds")
            
            return generated_titles
        except Exception as e:
            logger.error(f"Error in title generation: {str(e)}")
            raise

    def calculate_rouge_scores(self, reference_titles: List[str], generated_titles: List[str]) -> Dict[str, float]:
        try:
            if len(reference_titles) != len(generated_titles):
                raise ValueError(f"Mismatch in title counts: {len(reference_titles)} references vs {len(generated_titles)} generated")
                
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for ref_title, gen_title in zip(reference_titles, generated_titles):
                if not isinstance(ref_title, str) or not isinstance(gen_title, str):
                    continue
                    
                try:
                    scores = scorer.score(ref_title, gen_title)
                    
                    rouge1_scores.append(scores['rouge1'].fmeasure)
                    rouge2_scores.append(scores['rouge2'].fmeasure)
                    rougeL_scores.append(scores['rougeL'].fmeasure)
                except Exception as e:
                    logger.warning(f"Error calculating ROUGE score: {str(e)}")
            
            avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
            avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
            avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
            
            return {
                'rouge1': avg_rouge1,
                'rouge2': avg_rouge2,
                'rougeL': avg_rougeL,
                'average': (avg_rouge1 + avg_rouge2 + avg_rougeL) / 3
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}")
            raise

def main():
    try:
        generator = TitleGenerator()
        
        csv_path = '/content/test.csv'
        json_path = '/content/test.json'
        
        model_names = ['google/flan-t5-base', 'google/flan-t5-large']
        
        prompt_variations = [
            "Generate a concise title for the following article: ",
            "Create an appropriate title based on this text: "
        ]
        
        json_path = generator.convert_csv_to_json(csv_path, json_path)
        
        articles, reference_titles = generator.load_articles_from_json(json_path)
        logger.info(f"Loaded {len(articles)} articles with reference titles")
        
        results = {}
        
        for model_name in model_names:
            logger.info(f"\nProcessing model: {model_name}")
            model_results = {}
            
            for prompt in prompt_variations:
                logger.info(f"Using prompt: \"{prompt}\"")
                
                generated_titles = generator.generate_titles(
                    articles=articles,
                    model_name=model_name,
                    prompt_prefix=prompt
                )
                
                scores = generator.calculate_rouge_scores(reference_titles, generated_titles)
                model_results[prompt] = scores
                
                logger.info(f"ROUGE-1: {scores['rouge1']:.4f}")
                logger.info(f"ROUGE-2: {scores['rouge2']:.4f}")
                logger.info(f"ROUGE-L: {scores['rougeL']:.4f}")
                logger.info(f"Average: {scores['average']:.4f}")
            
            results[model_name] = model_results
        
        logger.info("\n===== FINAL ROUGE SCORES BY PROMPT =====")
        
        for prompt in prompt_variations:
            logger.info(f"\nPrompt: \"{prompt}\"")
            
            rouge1_total = sum(results[model][prompt]['rouge1'] for model in model_names)
            rouge2_total = sum(results[model][prompt]['rouge2'] for model in model_names)
            rougeL_total = sum(results[model][prompt]['rougeL'] for model in model_names)
            avg_total = sum(results[model][prompt]['average'] for model in model_names)
            
            for model in model_names:
                scores = results[model][prompt]
                logger.info(f"{model}:")
                logger.info(f"  ROUGE-1: {scores['rouge1']:.4f}")
                logger.info(f"  ROUGE-2: {scores['rouge2']:.4f}")
                logger.info(f"  ROUGE-L: {scores['rougeL']:.4f}")
                logger.info(f"  Average: {scores['average']:.4f}")
            
            model_count = len(model_names)
            logger.info(f"AVERAGE ACROSS MODELS:")
            logger.info(f"  ROUGE-1: {rouge1_total/model_count:.4f}")
            logger.info(f"  ROUGE-2: {rouge2_total/model_count:.4f}")
            logger.info(f"  ROUGE-L: {rougeL_total/model_count:.4f}")
            logger.info(f"  Average: {avg_total/model_count:.4f}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")

if __name__ == "__main__":
    main()