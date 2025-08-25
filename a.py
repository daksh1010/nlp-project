import json
import os
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pathos.multiprocessing import ProcessingPool
from collections import namedtuple

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

EXCLUDED_WORDS = set(stopwords.words('english'))
word_reducer = WordNetLemmatizer()
TextConfig = namedtuple('TextConfig', ['min_length', 'preserve_case'])

class TextMetrics:
    def __init__(self, content):
        self.content = content
        self.word_count = len(content.split())
        self.char_count = len(content)
    
    def get_density(self):
        return self.char_count / self.word_count if self.word_count else 0
    
    def get_complexity(self):
        total_len = sum(len(word) for word in self.content.split())
        return total_len / max(1, self.word_count)


def sanitize_content(text, is_heading=False, config=None):
    if config is None:
        config = TextConfig(min_length=4, preserve_case=False)
    
    if not config.preserve_case:
        text = text.lower()

    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())

    words = word_tokenize(text)
    filtered = []
    i = 0
    while i < len(words):
        w = words[i]
        if w.isalpha() and w.lower() not in EXCLUDED_WORDS and len(w) > 1:
            filtered.append(word_reducer.lemmatize(w.lower()))
        i += 1

    final = " ".join(filtered)
    return final if len(final.strip()) >= config.min_length else text.lower()


def process_entry(entry):
    try:
        return {
            'heading': sanitize_content(entry['title'], is_heading=True),
            'body': sanitize_content(entry['text'])
        }
    except Exception as error:
        print(f"Error in entry processing: {error}")
        return {'heading': '', 'body': ''}


def batch_process(entries, worker_count=None):
    if worker_count is None:
        worker_count = min(os.cpu_count(), 4)

    with ProcessingPool(worker_count) as pool:
        results = list(tqdm(
            pool.imap(process_entry, entries),
            total=len(entries),
            desc='Normalizing documents'
        ))
    return results


def split_train_validation(dataframe, val_count=500):
    index_order = np.random.permutation(len(dataframe))
    val_indices = set(index_order[:val_count])

    train_list, val_list = [], []
    idx = 0
    total = len(dataframe)

    pbar = tqdm(total=total, desc="Preparing training data")
    while idx < total:
        row = dataframe.iloc[idx]
        sample = {'title': row['title'], 'text': row['text']}
        if idx in val_indices:
            val_list.append(sample)
        else:
            train_list.append(sample)
        idx += 1
        pbar.update(1)
    pbar.close()

    return train_list, val_list


def prepare_datasets(source_train, source_test):
    train_samples, validation_samples = split_train_validation(source_train)

    test_samples = []
    idx = 0
    total = len(source_test)
    pbar = tqdm(total=total, desc="Preparing test data")
    while idx < total:
        row = source_test.iloc[idx]
        test_samples.append({'title': row['title'], 'text': row['text']})
        idx += 1
        pbar.update(1)
    pbar.close()

    return train_samples, test_samples, validation_samples


def analyze_corpus(corpus):
    word_counts = []
    char_counts = []

    idx = 0
    total = len(corpus)
    while idx < total:
        entry = corpus[idx]
        if 'body' in entry and entry['body']:
            stats = TextMetrics(entry['body'])
            word_counts.append(stats.word_count)
            char_counts.append(stats.char_count)
        idx += 1

    return {
        'avg_words': sum(word_counts) / max(1, len(word_counts)),
        'avg_chars': sum(char_counts) / max(1, len(char_counts)),
        'total_items': len(corpus)
    }


def execute_pipeline():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_data, test_data, val_data = prepare_datasets(train_df, test_df)

    processed_train = batch_process(train_data)
    processed_test = batch_process(test_data)
    processed_val = batch_process(val_data)

    output = {
        'train_corpus': processed_train,
        'test_corpus': processed_test,
        'validation_corpus': processed_val
    }

    with open('processed_corpus_without_punc.json', 'w') as out_file:
        json.dump(output, out_file, indent=4)

    stats = analyze_corpus(processed_train)
    print(stats)


if __name__ == '__main__':
    execute_pipeline()
