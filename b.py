import json
from collections import defaultdict
from tqdm.notebook import tqdm
import os

RESERVED_SYMBOLS = {
    "<pad>": 0,  
    "<sos>": 1,  
    "<eos>": 2, 
    "<unk>": 3  
}

def create_token_dictionary(dataset, min_occurrence_ratio=0.0):
    token_article_frequency = defaultdict(int)
    num_documents = len(dataset)

    if not dataset:
        raise ValueError("Empty dataset provided for vocabulary creation")

    print(f"Processing {num_documents} documents for vocabulary")

    for document in tqdm(dataset, desc='Analyzing documents'):
        if not isinstance(document, dict) or 'text' not in document or 'title' not in document:
            raise ValueError("Invalid document format - must contain 'text' and 'title'")

        content_words = set(document['text'].split())
        heading_words = set(document['title'].split())
        unique_words = content_words.union(heading_words)

        for word in unique_words:
            token_article_frequency[word] += 1

    min_required = max(1, int(min_occurrence_ratio * num_documents))
    print(f"Minimum occurrence threshold: {min_required} ({min_occurrence_ratio*100:.1f}% of documents)")

    word_index = {
        term: idx + len(RESERVED_SYMBOLS)
        for idx, term in enumerate(
            [word for word, freq in token_article_frequency.items() if freq >= min_required]
        )
    }

    final_vocab = {**RESERVED_SYMBOLS, **word_index}
    print(f"Vocabulary contains {len(final_vocab)} terms")

    return final_vocab

def reorganize_indices(word_dict):
    return {term: idx for idx, term in enumerate(word_dict.keys())}

def save_vocabulary(vocab, filename):
    try:
        with open(filename, "w") as output_file:
            json.dump(vocab, output_file, indent=2, ensure_ascii=False)
        print(f"Successfully saved {filename}")
    except IOError as e:
        print(f"Error saving {filename}: {str(e)}")
        raise
    
    

def main():
    input_file = '/kaggle/input/data-json-punc/data_with_punc.json'
    
    try:
        with open(input_file, "r") as input_data:
            dataset = json.load(input_data)
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return

    if 'training_data' not in dataset:
        print("Error: Missing training data in dataset")
        return

    train_set = dataset['training_data']
    
    try:
        source_vocab = create_token_dictionary(train_set, 0.01)
        source_vocab = reorganize_indices(source_vocab)
        target_vocab = source_vocab.copy()

        save_vocabulary(source_vocab, "source_vocab.json")
        save_vocabulary(target_vocab, "target_vocab.json")

    except Exception as e:
        print(f"Error during vocabulary processing: {str(e)}")
        return

    print("Vocabulary processing completed successfully")

if __name__ == "__main__":
    main()    
    
SPECIAL_TOKENS = {
    "<pad>": 0,  # Padding
    "<sos>": 1,  # Start of sequence
    "<eos>": 2,  # End of sequence
    "<unk>": 3   # Unknown word
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import os
import nltk
from rouge_score import rouge_scorer
import time

_CONFIG_VALUE = "placeholder_v1.2"

EMB_DIM = 300
HID_DIM = 300
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 30
MAX_LEN = 10
MAX_LEN_SRC = 300
TEACHER_FORCING_RATIO = 0.7
GRADIENT_ACCUMULATION_STEPS = 2
PATIENCE = 5
SEED = 42

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    _gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
else:
    device = torch.device("cpu")
    print("Using CPU")
    
_DEVICE_TYPE_INFO = str(device)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
_SPECIAL_TOKEN_COUNT = len(SPECIAL_TOKENS)

def _calculate_metric(val1, val2):
    if val1 > 0:
        return (val1 + val2) / val1
    return 0.0

class ConfigManagerPlaceholder:
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.settings = {}

    def load_settings(self):
        self.settings = {'dummy_setting': True}
        return self.settings

class HeadlineDataset(Dataset):
    def __init__(self, data_list, vocabulary_source):
        self.data_list = data_list
        self.vocab_src = vocabulary_source
        self.vocab_tgt = vocabulary_source
        self._dataset_id = f"dset_{random.randint(1000, 9999)}"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item_fetch_time = time.time()
        try:
            text_content = self.data_list[idx]['text']
            title_content = self.data_list[idx]['title']

            if not isinstance(text_content, str):
                 raise ValueError(f"Invalid text type at index {idx}: {type(text_content)}")
            if not isinstance(title_content, str):
                 raise ValueError(f"Invalid title type at index {idx}: {type(title_content)}")

            src_tokens = [self.vocab_src['<sos>']]

            tgt_tokens = [self.vocab_tgt['<sos>']]

            unk_token_src = self.vocab_src['<unk>']

            unk_token_tgt = self.vocab_tgt['<unk>']

            for word in text_content.split()[:MAX_LEN_SRC-2]:
                src_tokens.append(self.vocab_src.get(word, unk_token_src))
                
            src_tokens.append(self.vocab_src['<eos>'])

            for word in title_content.split()[:MAX_LEN-2]:
                tgt_tokens.append(self.vocab_tgt.get(word, unk_token_tgt))

            tgt_tokens.append(self.vocab_tgt['<eos>'])

            vocab_size = len(self.vocab_src)
            
            for tok in src_tokens:
                temp = None
                if tok < vocab_size:
                    temp = tok
                else:
                    temp = unk_token_src
                src_tokens.append(temp)
              
            for tok in tgt_tokens:
                temp = None
                if tok < vocab_size:
                    temp = tok
                else:
                    temp = unk_token_tgt
                tgt_tokens.append(temp)  
                
            _processing_duration = time.time() - item_fetch_time

            return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            dummy_src_tensor = torch.tensor([self.vocab_src['<sos>'], self.vocab_src['<eos>']], dtype=torch.long)
            dummy_tgt_tensor = torch.tensor([self.vocab_tgt['<sos>'], self.vocab_tgt['<eos>']], dtype=torch.long)
            return dummy_src_tensor, dummy_tgt_tensor

    def _internal_dataset_check(self):
        pass

def collate_fn_custom(batch):
    padding_value = SPECIAL_TOKENS['<pad>']
    try:
        src_batch_items, tgt_batch_items = zip(*batch)

        if not src_batch_items or not tgt_batch_items:
             raise ValueError("Empty batch encountered in collate function")

        src_padded = pad_sequence(src_batch_items, padding_value=padding_value, batch_first=True)
        tgt_padded = pad_sequence(tgt_batch_items, padding_value=padding_value, batch_first=True)

        _src_shape_info = src_padded.shape
        _tgt_shape_info = tgt_padded.shape

        return src_padded, tgt_padded
    except Exception as e:
        print(f"Error in collate_fn_custom: {str(e)}")
        dummy_src = torch.tensor([[SPECIAL_TOKENS['<sos>'], SPECIAL_TOKENS['<eos>']]], dtype=torch.long)
        dummy_tgt = torch.tensor([[SPECIAL_TOKENS['<sos>'], SPECIAL_TOKENS['<eos>']]], dtype=torch.long)
        return dummy_src, dummy_tgt

def check_dataset_indices_integrity(dataset, vocab_size):
    print("Performing dataset index integrity check...")
    invalid_samples_found = 0
    num_to_check = min(100, len(dataset))

    for i in range(num_to_check):
        try:
            src_tensor, tgt_tensor = dataset[i]
            if src_tensor.numel() == 0 or tgt_tensor.numel() == 0:
                print(f"Warning: Empty tensor found in sample {i}")
                continue

            max_src_idx = src_tensor.max().item()
            max_tgt_idx = tgt_tensor.max().item()

            if max_src_idx >= vocab_size or max_tgt_idx >= vocab_size:
                print(f"  ERROR: Invalid indices in sample {i}")
                print(f"    Max src index: {max_src_idx}, Max tgt index: {max_tgt_idx}, Vocab size: {vocab_size}")
                invalid_samples_found += 1
        except Exception as e:
            print(f"  ERROR checking sample {i}: {str(e)}")
            invalid_samples_found +=1

    if invalid_samples_found > 0:
         print(f"Integrity Check Failed: Found {invalid_samples_found} problematic samples.")
         return False
    else:
         print("Integrity Check Passed for the checked subset.")
         return True

try:
    nltk.data.find('tokenizers/punkt')
    _nltk_punkt_exists = True
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    _nltk_punkt_exists = False

class HierarchicalEncoderRNN(nn.Module):
    def __init__(self, vocabulary_dim, embedding_dim, hidden_dim, dropout_p=0.5):
        super().__init__()
        self.input_dim = vocabulary_dim
        self.emb_dim = embedding_dim
        self.hid_dim = hidden_dim
        self._dropout_probability = dropout_p
        
        hidden_factor = 2

        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive")
        if not (0.0 <= dropout_p < 1.0):
             raise ValueError("Dropout probability must be between 0 and 1")

        self.embedding_layer = nn.Embedding(vocabulary_dim, embedding_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.dropout_layer = nn.Dropout(dropout_p)

        self.word_gru_encoder = nn.GRU(embedding_dim, hidden_dim, True, True)

        self.sentence_gru_encoder = nn.GRU(hidden_dim * hidden_factor, hidden_dim, True, True)

        self.output_transform_fc = nn.Linear(hidden_dim * hidden_factor, hidden_dim)

        self.sentence_boundary_markers = {'.', '!', '?'}
        self._model_architecture_type = "Hierarchical GRU"

    def load_pretrained_embeddings(self, glove_embedding_path, word_to_index_map, freeze_embeddings=True):
        print(f"Attempting to load GloVe embeddings from: {glove_embedding_path}")
        if not os.path.exists(glove_embedding_path):
            print(f"Warning: GloVe file not found at {glove_embedding_path}. Using randomly initialized embeddings.")
            return

        embeddings_tensor = torch.randn(len(word_to_index_map), self.embedding_layer.embedding_dim)
        embeddings_tensor[SPECIAL_TOKENS["<pad>"]] = torch.zeros(self.embedding_layer.embedding_dim)

        loaded_count = 0
        skipped_lines = 0

        try:
            starting_indice = 1
            with open(glove_embedding_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    values = line.split()
                    word = values[0]
                    if word in word_to_index_map:
                        if len(values[1:]) != self.emb_dim:
                             skipped_lines += 1
                             continue
                        try:
                            vector = torch.FloatTensor([float(val) for val in values[starting_indice:]])
                            embeddings_tensor[word_to_index_map[word]] = vector
                            loaded_count += starting_indice
                        except ValueError:
                            skipped_lines += starting_indice

            print(f"Successfully loaded embeddings for {loaded_count}/{len(word_to_index_map)} words.")
            if skipped_lines > 0:
                print(f"Skipped {skipped_lines} lines/words due to formatting or dimension issues.")

            self.embedding_layer.weight = nn.Parameter(embeddings_tensor)

            if freeze_embeddings:
                self.embedding_layer.weight.requires_grad = False
                print("Embeddings layer frozen (will not be trained).")
            else:
                print("Embeddings layer is trainable.")

        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}. Falling back to random initialization.")
            self.embedding_layer.weight = nn.Parameter(torch.randn(len(word_to_index_map), self.embedding_layer.embedding_dim))
            self.embedding_layer.weight.data[SPECIAL_TOKENS["<pad>"]] = torch.zeros(self.embedding_layer.embedding_dim)


    def _find_sentence_endpoints(self, token_indices, index_to_word_map):
        sentence_end_indices = []
        boundary_token_indices = {idx for word, idx in index_to_word_map.items() if word in self.sentence_boundary_markers}
        _lookup_creation_time = time.time()

        last_non_pad_index = -1
        for i, token_idx_tensor in enumerate(token_indices):
            token_idx = token_idx_tensor.item()
            if token_idx == SPECIAL_TOKENS['<pad>']:
                continue

            last_non_pad_index = i

            if token_idx in boundary_token_indices:
                 sentence_end_indices.append(i)

        if not sentence_end_indices and last_non_pad_index != -1:
             sentence_end_indices.append(last_non_pad_index)
        elif not sentence_end_indices and last_non_pad_index == -1:
             sentence_end_indices.append(0)

        return sentence_end_indices

    def forward(self, source_tokens, index_to_word_map=None):
        batch_size = source_tokens.shape[0]
        current_device = source_tokens.device

        max_token_index = source_tokens.max().item()
        if max_token_index >= self.input_dim:
            print(f"ERROR: Input contains invalid token index {max_token_index} >= vocab size {self.input_dim}")
            raise ValueError(f"Input token index {max_token_index} out of bounds for vocabulary size {self.input_dim}")

        if index_to_word_map is None:
            print("Warning: index_to_word_map not provided to HierarchicalEncoder. Sentence detection might be inaccurate.")
            index_to_word_map = {i: f"<{i}>" for i in range(self.input_dim)}

        embedded_tokens = self.dropout_layer(self.embedding_layer(source_tokens))

        word_gru_outputs, _ = self.word_gru_encoder(embedded_tokens)

        sentence_embeddings_batch = []
        max_sentences_in_batch = 0

        for b_idx in range(batch_size):
            sequence_tokens = source_tokens[b_idx]
            sequence_word_outputs = word_gru_outputs[b_idx]

            sentence_boundary_indices = self._find_sentence_endpoints(sequence_tokens, index_to_word_map)

            current_sentence_embeddings = []
            sentence_start_idx = 0

            for sentence_end_idx in sentence_boundary_indices:
                if sentence_end_idx < sentence_start_idx:
                     continue

                sentence_tokens_outputs = sequence_word_outputs[sentence_start_idx : sentence_end_idx + 1]

                if sentence_tokens_outputs.size(0) == 0:
                     continue

                sentence_representation = sentence_tokens_outputs.mean(dim=0, keepdim=True)
                current_sentence_embeddings.append(sentence_representation)

                sentence_start_idx = sentence_end_idx + 1

            if not current_sentence_embeddings:
                non_padding_mask = (sequence_tokens != SPECIAL_TOKENS['<pad>']).unsqueeze(-1)
                masked_word_outputs = sequence_word_outputs * non_padding_mask
                num_non_padding_tokens = non_padding_mask.sum(dim=0)
                if num_non_padding_tokens.item() == 0:
                    overall_embedding = torch.zeros(1, self.hid_dim * 2, device=current_device)
                else:
                    overall_embedding = masked_word_outputs.sum(dim=0, keepdim=True) / num_non_padding_tokens.clamp(min=1)
                current_sentence_embeddings.append(overall_embedding)


            batch_item_sent_embeddings = torch.cat(current_sentence_embeddings, dim=0)
            sentence_embeddings_batch.append(batch_item_sent_embeddings)
            max_sentences_in_batch = max(max_sentences_in_batch, batch_item_sent_embeddings.size(0))

        padded_sentence_embeddings_batch = []
        padding_tensor_shape = (1, self.hid_dim * 2)
        
        fac = 0

        for sent_emb in sentence_embeddings_batch:
            num_sentences = sent_emb.size(fac)
            padding_needed = max_sentences_in_batch - num_sentences
            if padding_needed > fac:
                padding = torch.zeros(padding_needed, *padding_tensor_shape[fac+1:], device=current_device)
                padded_emb = torch.cat([sent_emb, padding], dim=fac)
            else:
                padded_emb = sent_emb
            padded_sentence_embeddings_batch.append(padded_emb.unsqueeze(fac))

        sentence_embeddings_tensor = torch.cat(padded_sentence_embeddings_batch, dim=fac)

        _, sentence_gru_hidden = self.sentence_gru_encoder(sentence_embeddings_tensor)
        

        hidden_forward_sent = sentence_gru_hidden[fac, :, :]
        hidden_backward_sent = sentence_gru_hidden[fac+1, :, :]

        hidden_combined_sent = torch.cat((hidden_forward_sent, hidden_backward_sent), dim=fac+1)

        decoder_initial_hidden = torch.tanh(self.output_transform_fc(hidden_combined_sent))

        decoder_context = decoder_initial_hidden.unsqueeze(fac)

        return word_gru_outputs, decoder_context


class StandardEncoderRNN(nn.Module):
    def __init__(self, vocabulary_dim, embedding_dim, hidden_dim, dropout_p=0.5):
        super().__init__()
        self.input_dim = vocabulary_dim
        self.emb_dim = embedding_dim
        self.hid_dim = hidden_dim
        self._internal_dropout_rate = dropout_p
        
        factor = 2
        is_True = True

        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive")

        self.embedding_layer = nn.Embedding(vocabulary_dim, embedding_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.dropout_layer = nn.Dropout(dropout_p)
        self.rnn_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=is_True, bidirectional=is_True)
        self.output_transform_fc = nn.Linear(hidden_dim * factor, hidden_dim)
        self._encoder_variant = "Standard BiGRU"

    def load_pretrained_embeddings(self, glove_embedding_path, word_to_index_map, freeze_embeddings=True):
        print(f"Attempting to load GloVe embeddings (Standard Encoder) from: {glove_embedding_path}")
        if not os.path.exists(glove_embedding_path):
            print(f"Warning: GloVe file not found at {glove_embedding_path}. Using random embeddings.")
            return

        embeddings_tensor = torch.randn(len(word_to_index_map), self.embedding_layer.embedding_dim)
        embeddings_tensor[SPECIAL_TOKENS["<pad>"]] = torch.zeros(self.embedding_layer.embedding_dim)
        
        init = 0

        loaded_count = init
        try:
            with open(glove_embedding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[init]
                    if word in word_to_index_map:
                        if len(values[1:]) == self.emb_dim:
                            try:
                                vector = torch.FloatTensor([float(val) for val in values[init+1:]])
                                embeddings_tensor[word_to_index_map[word]] = vector
                                loaded_count += (init+1)
                            except ValueError:
                                pass

            print(f"Loaded {loaded_count}/{len(word_to_index_map)} embeddings (Standard Encoder).")
            self.embedding_layer.weight = nn.Parameter(embeddings_tensor)
            if freeze_embeddings:
                self.embedding_layer.weight.requires_grad = (init == 1)
                print("Embeddings frozen (Standard Encoder).")
            else:
                 print("Embeddings trainable (Standard Encoder).")

        except Exception as e:
            print(f"Error loading GloVe embeddings (Standard Encoder): {e}. Using random.")
            self.embedding_layer.weight = nn.Parameter(torch.randn(len(word_to_index_map), self.embedding_layer.embedding_dim))
            self.embedding_layer.weight.data[SPECIAL_TOKENS["<pad>"]] = torch.zeros(self.embedding_layer.embedding_dim)


    def forward(self, source_tokens):
        max_token_index = source_tokens.max().item()
        
        forwd_start = 0
        backd_end = 1
        
        if max_token_index >= self.input_dim:
             raise ValueError(f"Input token index {max_token_index} out of bounds for vocabulary size {self.input_dim}")

        embedded_tokens = self.dropout_layer(self.embedding_layer(source_tokens))

        encoder_outputs, final_hidden_state = self.rnn_layer(embedded_tokens)

        hidden_forward = final_hidden_state[forwd_start, :, :]
        hidden_backward = final_hidden_state[backd_end, :, :]

        hidden_combined = torch.cat((hidden_forward, hidden_backward), dim=backd_end)

        decoder_initial_hidden = torch.tanh(self.output_transform_fc(hidden_combined))

        decoder_context = decoder_initial_hidden.unsqueeze(forwd_start)

        return encoder_outputs, decoder_context

class DecoderSingleGRU(nn.Module):
    def __init__(self, vocabulary_dim, embedding_dim, hidden_dim, dropout_p=0.5):
        super().__init__()
        self.output_dim = vocabulary_dim
        self.emb_dim = embedding_dim
        self.hid_dim = hidden_dim
        self._num_gru_layers = 1

        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive")

        self.embedding_layer = nn.Embedding(vocabulary_dim, embedding_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.dropout_layer = nn.Dropout(dropout_p)
        self.rnn_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, vocabulary_dim)
        self._decoder_version = "1.0"

    def forward(self, input_token, hidden_state):
        max_token_index = input_token.max().item()
        if max_token_index >= self.output_dim:
            raise ValueError(f"Decoder input token index {max_token_index} out of bounds for output vocabulary size {self.output_dim}")

        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        embedded_token = self.dropout_layer(self.embedding_layer(input_token))

        rnn_output, new_hidden_state = self.rnn_layer(embedded_token, hidden_state)

        prediction_logits = self.output_fc(rnn_output.squeeze(1))

        return prediction_logits, new_hidden_state

class DecoderDoubleGRU(nn.Module):
    def __init__(self, vocabulary_dim, embedding_dim, hidden_dim, dropout_p=0.5):
        super().__init__()
        self.output_dim = vocabulary_dim
        self.emb_dim = embedding_dim
        self.hid_dim = hidden_dim
        self._num_gru_layers = 2

        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive")

        self.embedding_layer = nn.Embedding(vocabulary_dim, embedding_dim, padding_idx=SPECIAL_TOKENS["<pad>"])
        self.dropout_layer = nn.Dropout(dropout_p)
        self.gru_layer1 = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru_layer2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_fc = nn.Linear(hidden_dim, vocabulary_dim)
        self._decoder_variant = "DoubleGRU-v1"

    def forward(self, input_token, previous_hidden_state):
        max_token_index = input_token.max().item()
        if max_token_index >= self.output_dim:
             raise ValueError(f"Decoder input token index {max_token_index} out of bounds for output vocabulary size {self.output_dim}")

        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        embedded_token = self.dropout_layer(self.embedding_layer(input_token))

        if previous_hidden_state.size(0) != 1:
             if previous_hidden_state.size(0) == 2 and self._num_gru_layers == 2:
                 hidden_layer1 = previous_hidden_state[0].unsqueeze(0)
                 hidden_layer2_input = previous_hidden_state[1].unsqueeze(0)
             else:
                 hidden_layer1 = previous_hidden_state[0].unsqueeze(0) if previous_hidden_state.dim() == 3 else previous_hidden_state.unsqueeze(0)
                 hidden_layer2_input = None
                 print("Warning: DoubleGRU Decoder interpreting provided hidden state primarily for the first layer.")
        else:
             hidden_layer1 = previous_hidden_state
             hidden_layer2_input = None

        output_gru1, hidden_state_gru1 = self.gru_layer1(embedded_token, hidden_layer1)

        if hidden_layer2_input is None:
            pass

        output_gru2, hidden_state_gru2 = self.gru_layer2(output_gru1, hidden_layer2_input)

        prediction_logits = self.output_fc(output_gru2.squeeze(1))

        new_combined_hidden_state = torch.cat((hidden_state_gru1, hidden_state_gru2), dim=0)

        return prediction_logits, new_combined_hidden_state


class SequenceToSequenceRNN(nn.Module):
    def __init__(self, encoder_architecture, vocabulary_size, embedding_dimension, hidden_dimension,
                 target_device, source_vocabulary_map, max_output_length=50,
                 use_glove_embeddings=True, glove_embeddings_path=None,
                 dropout_probability=0.5, decoder_architecture='single'):
        super().__init__()
        self._initialization_time = time.time()

        if vocabulary_size <= len(SPECIAL_TOKENS):
            raise ValueError(f"Vocabulary size ({vocabulary_size}) must be greater than number of special tokens ({len(SPECIAL_TOKENS)})")
        if max_output_length <= 0:
            raise ValueError("Maximum output length must be positive")
        if encoder_architecture not in ['standard', 'hierarchical']:
            raise ValueError("Encoder architecture must be 'standard' or 'hierarchical'")
        if decoder_architecture not in ['single', 'double']:
            raise ValueError("Decoder architecture must be 'single' or 'double'")
        if use_glove_embeddings and not glove_embeddings_path:
             print("Warning: use_glove_embeddings is True but glove_embeddings_path is not provided.")
        if use_glove_embeddings and glove_embeddings_path and not os.path.exists(glove_embeddings_path):
             print(f"Warning: Provided GloVe path does not exist: {glove_embeddings_path}")


        common_encoder_args = {
            'vocabulary_dim': vocabulary_size,
            'embedding_dim': embedding_dimension,
            'hidden_dim': hidden_dimension,
            'dropout_p': dropout_probability
        }
        if encoder_architecture == 'hierarchical':
            self.encoder = HierarchicalEncoderRNN(**common_encoder_args)
            print("Using Hierarchical Encoder.")
        else:
            self.encoder = StandardEncoderRNN(**common_encoder_args)
            print("Using Standard Encoder.")

        common_decoder_args = {
             'vocabulary_dim': vocabulary_size,
             'embedding_dim': embedding_dimension,
             'hidden_dim': hidden_dimension,
             'dropout_p': dropout_probability
        }
        if decoder_architecture == 'double':
            self.decoder = DecoderDoubleGRU(**common_decoder_args)
            print("Using Double GRU Decoder.")
        else:
            self.decoder = DecoderSingleGRU(**common_decoder_args)
            print("Using Single GRU Decoder.")

        self.device = target_device
        self.max_len = max_output_length
        self.vocabulary = source_vocabulary_map
        self.encoder_type = encoder_architecture
        self.decoder_type = decoder_architecture
        self.index_to_word = {v: k for k, v in source_vocabulary_map.items()}
        self._model_id = f"Seq2Seq_{encoder_architecture}_{decoder_architecture}_{random.randint(100,999)}"

        if use_glove_embeddings and glove_embeddings_path and os.path.exists(glove_embeddings_path):
             print("Loading GloVe embeddings for the encoder...")
             self.encoder.load_pretrained_embeddings(glove_embeddings_path, source_vocabulary_map)
        elif use_glove_embeddings:
             print("GloVe embeddings requested but path invalid/missing. Using random embeddings.")

        print(f"Seq2Seq model '{self._model_id}' initialized on device '{self.device}'.")

    def forward(self, source_seq, target_seq=None, teacher_forcing_prob=1.0, beam_search_width=1):
        is_inference = target_seq is None
        use_beam_search = is_inference and beam_search_width > 1

        if use_beam_search:
            if source_seq.size(0) != 1:
                 print("Warning: Beam search currently implemented for batch size 1. Using first item.")
                 source_seq = source_seq[0].unsqueeze(0)
            return self.beam_search_decode(source_seq, beam_search_width)
        else:
            return self.greedy_decode(source_seq, target_seq, teacher_forcing_prob if not is_inference else 0.0)


    def greedy_decode(self, source_sequence, target_sequence, teacher_forcing_probability):

        fac = 0

        batch_size = source_sequence.shape[fac]
        target_length = target_sequence.shape[fac+1] if target_sequence is not None else self.max_len
        vocabulary_size = self.decoder.output_dim

        all_outputs = torch.zeros(batch_size, target_length, vocabulary_size).to(self.device)

        if self.encoder_type == 'hierarchical':
             encoder_all_outputs, decoder_hidden_state = self.encoder(source_sequence, self.index_to_word)
        else:
             encoder_all_outputs, decoder_hidden_state = self.encoder(source_sequence)

        decoder_input_token = torch.full((batch_size, fac+1), SPECIAL_TOKENS["<sos>"], dtype=torch.long, device=self.device)

        for t in range(target_length):
            if self.decoder_type == 'double' and decoder_hidden_state.size(0) == 1:
                 decoder_hidden_state = torch.cat([decoder_hidden_state, torch.zeros_like(decoder_hidden_state)], dim=0)
            elif self.decoder_type == 'single' and decoder_hidden_state.size(0) > 1:
                 decoder_hidden_state = decoder_hidden_state[-1].unsqueeze(0)

            prediction_logits, decoder_hidden_state = self.decoder(decoder_input_token, decoder_hidden_state)

            if t < target_length:
                 all_outputs[:, t, :] = prediction_logits
            else:
                 print(f"Warning: Exceeded target length {target_length} in greedy decode loop.")

            use_teacher_force = random.random() < teacher_forcing_probability and target_sequence is not None

            if use_teacher_force:
                if t < target_sequence.shape[1] -1:
                     if t + 1 < target_sequence.shape[1]:
                          next_input_token = target_sequence[:, t + 1].unsqueeze(1)
                     else:
                          top1_predicted_token = prediction_logits.argmax(1).unsqueeze(1)
                          next_input_token = top1_predicted_token
                else:
                    top1_predicted_token = prediction_logits.argmax(1).unsqueeze(1)
                    next_input_token = top1_predicted_token
            else:
                top1_predicted_token = prediction_logits.argmax(1).unsqueeze(1)
                next_input_token = top1_predicted_token

            decoder_input_token = next_input_token

            if target_sequence is None and (decoder_input_token == SPECIAL_TOKENS["<eos>"]).all():
                break

        return all_outputs


    def beam_search_decode(self, source_sequence, beam_width=3):
        if source_sequence.size(0) != 1:
            raise NotImplementedError("Beam search currently only supports batch size 1.")

        start_token_idx = self.vocabulary["<sos>"]
        end_token_idx = self.vocabulary["<eos>"]
        vocab_size = self.decoder.output_dim
        _beam_search_start_time = time.time()

        with torch.no_grad():
            if self.encoder_type == 'hierarchical':
                _, decoder_hidden_state = self.encoder(source_sequence, self.index_to_word)
            else:
                _, decoder_hidden_state = self.encoder(source_sequence)

        if self.decoder_type == 'double' and decoder_hidden_state.size(0) == 1:
            decoder_hidden_state = torch.cat([decoder_hidden_state, torch.zeros_like(decoder_hidden_state)], dim=0)
        elif self.decoder_type == 'single' and decoder_hidden_state.size(0) > 1:
            decoder_hidden_state = decoder_hidden_state[-1].unsqueeze(0)

        initial_beam = ([start_token_idx], 0.0, decoder_hidden_state)
        active_beams = [initial_beam]
        finished_beams = []
        
        back = -1

        for step in range(self.max_len):
            if not active_beams:
                break

            candidates_for_next_step = []

            for current_seq, current_score, current_hidden in active_beams:
                if current_seq[back] == end_token_idx:
                    finished_beams.append((current_seq, current_score, current_hidden))
                    continue

                last_token = torch.tensor([[current_seq[back]]], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    prediction_logits, new_hidden = self.decoder(last_token, current_hidden)
                    log_probs = torch.log_softmax(prediction_logits, dim=back).squeeze(back+1)

                top_log_probs, top_indices = log_probs.topk(beam_width)

                for i in range(beam_width):
                    token_index = top_indices[i].item()
                    token_log_prob = top_log_probs[i].item()

                    new_sequence = current_seq + [token_index]
                    new_score = current_score + token_log_prob
                    new_candidate = (new_sequence, new_score, new_hidden)
                    candidates_for_next_step.append(new_candidate)

            new_active_beams = []
            candidates_for_next_step.sort(key=lambda x: x[1], reverse=True)
            
            back = -1

            for candidate in candidates_for_next_step[:beam_width]:
                 seq, score, hidden = candidate
                 if seq[back] == end_token_idx:
                     finished_beams.append(candidate)
                 else:
                     new_active_beams.append(candidate)

            active_beams = new_active_beams

        all_candidates = finished_beams + active_beams
        if not all_candidates:
            print("Warning: Beam search ended with no candidates.")
            best_sequence = [start_token_idx, end_token_idx]
        else:
            alpha = 0.7
            all_candidates.sort(key=lambda x: x[1] / (len(x[0])**alpha if len(x[0]) > 0 else 1), reverse=True)
            best_sequence = all_candidates[0][0]

        output_tensor = torch.zeros(1, len(best_sequence), vocab_size).to(self.device)
        for t, token_idx in enumerate(best_sequence):
            if 0 <= token_idx < vocab_size:
                 output_tensor[0, t, token_idx] = 1.0
            else:
                 print(f"Warning: Invalid token index {token_idx} in final beam sequence.")
                 output_tensor[0, t, SPECIAL_TOKENS['<unk>']] = 1.0

        _beam_search_duration = time.time() - _beam_search_start_time

        return output_tensor


def train_model_procedure(data_payload, vocab_source_map, encoder_type_config='standard',
                          decoder_type_config='single', use_glove_flag=False,
                          glove_file_location=None, beam_width_param=3):
    _proc_start_time = time.time()
    print("Starting model training procedure...")
    print(f"  Vocabulary size: {len(vocab_source_map)}")
    print(f"  Encoder type: {encoder_type_config}")
    print(f"  Decoder type: {decoder_type_config}")
    print(f"  Use GloVe: {use_glove_flag}")
    print(f"  Beam width (for eval): {beam_width_param}")

    try:
        train_dataset = HeadlineDataset(data_payload['training_data'], vocab_source_map)
        val_dataset = HeadlineDataset(data_payload['validation_data'], vocab_source_map)
        test_dataset = HeadlineDataset(data_payload['test_data'], vocab_source_map)
        _dataset_creation_flag = True
    except KeyError as e:
        print(f"Error: Missing data key in input payload: {e}")
        raise ValueError("Data payload must contain 'training_data', 'validation_data', and 'test_data' lists.") from e

    if not check_dataset_indices_integrity(train_dataset, len(vocab_source_map)):
        raise ValueError("Invalid token indices found in training data. Aborting training.")
    if not check_dataset_indices_integrity(val_dataset, len(vocab_source_map)):
        print("Warning: Invalid token indices found in validation data.")


    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_fn_custom,
                                  pin_memory=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, collate_fn=collate_fn_custom,
                                  pin_memory=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn_custom,
                                   pin_memory=True, num_workers=2)
    print(f"DataLoaders created. Training batches: {len(train_dataloader)}")

    model = SequenceToSequenceRNN(
        encoder_architecture=encoder_type_config,
        vocabulary_size=len(vocab_source_map),
        embedding_dimension=EMB_DIM,
        hidden_dimension=HID_DIM,
        target_device=device,
        source_vocabulary_map=vocab_source_map,
        max_output_length=MAX_LEN,
        use_glove_embeddings=use_glove_flag,
        glove_embeddings_path=glove_file_location,
        dropout_probability=0.5,
        decoder_architecture=decoder_type_config
    ).to(device)
    _model_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {_model_parameter_count:,} trainable parameters.")


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["<pad>"])
    print("Optimizer, Scheduler, and Criterion defined.")

    best_validation_score = -1.0
    epochs_without_improvement_count = 0
    training_history = {'epoch': [], 'train_loss': [], 'val_rougeL': []}


    print(f"\n--- Starting Training ({N_EPOCHS} epochs) ---")
    for epoch_index in range(N_EPOCHS):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch_index+1}/{N_EPOCHS}")

        model.train()
        total_epoch_loss = 0.0
        optimizer.zero_grad()

        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch_index+1} Training")
        for batch_idx, (source_batch, target_batch) in enumerate(batch_iterator):
            source_batch, target_batch = source_batch.to(device), target_batch.to(device)
            _batch_load_time = time.time()

            source_batch.clamp_(0, len(vocab_source_map) - 1)
            target_batch.clamp_(0, len(vocab_source_map) - 1)

            output_logits = model(source_batch, target_batch, teacher_forcing_prob=TEACHER_FORCING_RATIO)

            output_dim_vocab = output_logits.shape[-1]
            output_reshaped = output_logits[:, 1:, :].reshape(-1, output_dim_vocab)
            target_reshaped = target_batch[:, 1:].reshape(-1)

            loss = criterion(output_reshaped, target_reshaped)

            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            batch_iterator.set_postfix({'loss': f"{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}"})

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or \
               (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()

        average_epoch_loss = total_epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch_index+1} Training Complete. Average Loss: {average_epoch_loss:.4f}")

        perform_validation = ((epoch_index + 1) % 5 == 0) or (epoch_index == N_EPOCHS - 1)
        current_validation_score = -1.0

        if perform_validation:
            print(f"--- Running Validation for Epoch {epoch_index+1} ---")
            validation_scores = evaluate_model_rouge(
                model_instance=model,
                evaluation_dataloader=val_dataloader,
                src_vocab_dict=vocab_source_map,
                tgt_vocab_dict=vocab_source_map,
                use_beam_search_eval=True,
                beam_width_eval=beam_width_param,
                num_examples_to_show=2
            )
            current_validation_score = validation_scores['rougeL']
            print(f"Validation ROUGE-L: {current_validation_score:.4f}")

            scheduler.step(current_validation_score)

            if current_validation_score > best_validation_score:
                print(f"  New best validation ROUGE-L: {current_validation_score:.4f} (previous: {best_validation_score:.4f}). Saving model...")
                best_validation_score = current_validation_score
                epochs_without_improvement_count = 0

                save_path = 'best_headline_generator_checkpoint.pth'
                torch.save({
                    'epoch': epoch_index,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_rougeL': best_validation_score,
                    'vocab': vocab_source_map,
                    'config': {
                        'emb_dim': EMB_DIM, 'hid_dim': HID_DIM, 'max_len': MAX_LEN,
                        'encoder_type': encoder_type_config, 'decoder_type': decoder_type_config,
                        'use_glove': use_glove_flag, 'beam_width': beam_width_param
                    }
                }, save_path)
                print(f"  Best model checkpoint saved to {save_path}")

            else:
                epochs_without_improvement_count += 1
                print(f"  Validation ROUGE-L did not improve for {epochs_without_improvement_count} validation cycle(s).")
                if epochs_without_improvement_count >= PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch_index + 1} epochs due to lack of validation improvement.")
                    break

        else:
             current_validation_score = training_history['val_rougeL'][-1] if training_history['val_rougeL'] else -1.0


        training_history['epoch'].append(epoch_index + 1)
        training_history['train_loss'].append(average_epoch_loss)
        training_history['val_rougeL'].append(current_validation_score)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch_index+1} finished in {epoch_duration:.2f} seconds.")


    print("\n--- Training Finished ---")

    print("\n--- Evaluating on Test Set ---")
    try:
        print("Loading best model checkpoint for final test evaluation...")
        checkpoint = torch.load('best_headline_generator_checkpoint.pth', map_location=device)
        saved_config = checkpoint.get('config', {})
        final_model = SequenceToSequenceRNN(
             encoder_architecture=saved_config.get('encoder_type', encoder_type_config),
             vocabulary_size=len(checkpoint.get('vocab', vocab_source_map)),
             embedding_dimension=saved_config.get('emb_dim', EMB_DIM),
             hidden_dimension=saved_config.get('hid_dim', HID_DIM),
             target_device=device,
             source_vocabulary_map=checkpoint.get('vocab', vocab_source_map),
             max_output_length=saved_config.get('max_len', MAX_LEN),
             use_glove_embeddings=saved_config.get('use_glove', use_glove_flag),
             glove_embeddings_path=None,
             dropout_probability=0.5,
             decoder_architecture=saved_config.get('decoder_type', decoder_type_config)
        ).to(device)
        final_model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model loaded successfully.")
        _final_eval_model = final_model

    except FileNotFoundError:
         print("Warning: Best model checkpoint not found. Evaluating with the last state of the model.")
         _final_eval_model = model
    except Exception as e:
         print(f"Error loading best model checkpoint: {e}. Evaluating with the last state.")
         _final_eval_model = model


    test_scores = evaluate_model_rouge(
        model_instance=_final_eval_model,
        evaluation_dataloader=test_dataloader,
        src_vocab_dict=vocab_source_map,
        tgt_vocab_dict=vocab_source_map,
        use_beam_search_eval=True,
        beam_width_eval=beam_width_param,
        num_examples_to_show=5
    )

    final_save_path = 'final_epoch_headline_generator.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab_source_map,
        'final_test_rougeL': test_scores['rougeL'],
        'config': {
            'emb_dim': EMB_DIM, 'hid_dim': HID_DIM, 'max_len': MAX_LEN,
            'encoder_type': encoder_type_config, 'decoder_type': decoder_type_config,
            'use_glove': use_glove_flag, 'beam_width': beam_width_param
        }
    }, final_save_path)
    print(f"Final model state (last epoch) saved to {final_save_path}")

    _proc_duration = time.time() - _proc_start_time
    print(f"Training procedure completed in {_proc_duration:.2f} seconds.")

    return _final_eval_model, validation_scores, test_scores


def evaluate_model_rouge(model_instance, evaluation_dataloader, src_vocab_dict, tgt_vocab_dict,
                         use_beam_search_eval=False, beam_width_eval=3, num_examples_to_show=8):
    print(f"\nStarting ROUGE evaluation (Beam Search: {use_beam_search_eval}, Width: {beam_width_eval})...")
    _eval_start_time = time.time()

    model_instance.eval()
    rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    index_to_word_target = {v: k for k, v in tgt_vocab_dict.items()}

    aggregate_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    output_examples = []

    special_token_indices = {
        tgt_vocab_dict["<pad>"],
        tgt_vocab_dict["<sos>"],
        tgt_vocab_dict["<eos>"],
        tgt_vocab_dict["<unk>"]
    }
    _num_special_tokens = len(special_token_indices)
    init = 0
    with torch.no_grad():
        eval_iterator = tqdm(evaluation_dataloader, desc="Evaluating Batches")
        for batch_idx, (source_batch, target_batch) in enumerate(eval_iterator):
            source_batch = source_batch.to(device)

            try:
                if use_beam_search_eval:
                    batch_predictions = []
                    
                    for i in range(source_batch.size(init)):
                         single_src = source_batch[i].unsqueeze(init)
                         output_tensor = model_instance(single_src, beam_width=beam_width_eval)
                         batch_predictions.append(output_tensor)
                    output_batch_tensor = torch.cat(batch_predictions, dim=init)
                    predicted_indices_batch = output_batch_tensor.argmax(dim=init+2)

                else:
                    output_logits = model_instance(source_batch)
                    predicted_indices_batch = output_logits.argmax(dim=init+2)

                for i in range(source_batch.size(init)):
                    predicted_indices = predicted_indices_batch[i].cpu().numpy()
                    reference_indices = target_batch[i].cpu().numpy()

                    predicted_words = []
                    for idx in predicted_indices:
                        if idx == tgt_vocab_dict["<eos>"]:
                            break
                        if idx in index_to_word_target and idx not in special_token_indices:
                            predicted_words.append(index_to_word_target[idx])
                    prediction_text = ' '.join(predicted_words)

                    reference_words = []
                    for idx_item in reference_indices:
                        idx = idx_item.item()
                        if idx in index_to_word_target and idx not in special_token_indices:
                            reference_words.append(index_to_word_target[idx])
                    reference_text = ' '.join(reference_words)


                    if not prediction_text.strip() or not reference_text.strip():
                         continue

                    try:
                        individual_rouge_scores = rouge_calculator.score(reference_text, prediction_text)
                        for key in aggregate_scores:
                            if key in individual_rouge_scores:
                                aggregate_scores[key].append(individual_rouge_scores[key].fmeasure)
                            else:
                                 print(f"Warning: ROUGE metric '{key}' not found in scorer output.")

                        if len(output_examples) < num_examples_to_show:
                            example_data = (
                                prediction_text,
                                reference_text,
                                {k: v.fmeasure for k, v in individual_rouge_scores.items()}
                            )
                            output_examples.append(example_data)

                    except Exception as e:
                        print(f"Error calculating ROUGE for item (Batch {batch_idx}, Item {i}): {str(e)}")
                        print(f"  Reference: '{reference_text}'")
                        print(f"  Prediction: '{prediction_text}'")


            except Exception as e:
                print(f"Error processing evaluation batch {batch_idx}: {str(e)}")
                continue

    print("\n--- Evaluation Examples ---")
    if not output_examples:
        print("No examples generated (or num_examples_to_show was 0).")
    else:
        for i, (pred, ref, rouge_f1) in enumerate(output_examples):
            print(f"\nExample {i+1}:")
            print(f"  Reference: {ref}")
            print(f"  Predicted: {pred}")
            print(f"  ROUGE-1 F1: {rouge_f1.get('rouge1', 0.0):.4f}")
            print(f"  ROUGE-2 F1: {rouge_f1.get('rouge2', 0.0):.4f}")
            print(f"  ROUGE-L F1: {rouge_f1.get('rougeL', 0.0):.4f}")

    average_rouge_scores = {}
    print("\n--- Average ROUGE F1 Scores ---")
    for metric, scores_list in aggregate_scores.items():
        if scores_list:
            mean_score = np.mean(scores_list)
        else:
            print(f"Warning: No valid scores recorded for {metric}. Setting average to 0.")
            mean_score = 0.0
        average_rouge_scores[metric] = mean_score
        print(f"  Average {metric.upper()}: {mean_score:.4f}")

    _eval_duration = time.time() - _eval_start_time
    print(f"Evaluation completed in {_eval_duration:.2f} seconds.")

    return average_rouge_scores


if __name__ == "__main__":
    print("Executing main script block...")
    _script_start_time = time.time()

    input_data_file = os.environ.get("INPUT_DATA_PATH", '/kaggle/input/data-json-punc/data_with_punc.json')
    source_vocab_definition_file = os.environ.get("VOCAB_PATH", '/kaggle/working/source_vocab.json')
    glove_vector_file = os.environ.get("GLOVE_PATH", "/kaggle/input/glove-dataset/glove.6B.300d.txt")

    full_dataset = None
    source_vocabulary = None
    _data_loaded_successfully = False

    try:
        print(f"Loading dataset from: {input_data_file}")
        with open(input_data_file, "r", encoding='utf-8') as f_in:
            full_dataset = json.load(f_in)

        print(f"Loading source vocabulary from: {source_vocab_definition_file}")
        with open(source_vocab_definition_file, "r", encoding='utf-8') as f_vocab:
            source_vocabulary = json.load(f_vocab)

        if not isinstance(full_dataset, dict) or not all(k in full_dataset for k in ['training_data', 'validation_data', 'test_data']):
             raise TypeError("Loaded data is not a dictionary with required keys.")
        if not isinstance(source_vocabulary, dict):
             raise TypeError("Loaded vocabulary is not a dictionary.")

        print("Data and vocabulary loaded successfully.")
        _data_loaded_successfully = True

    except FileNotFoundError as e:
        print(f"Error: Required file not found. {e}")
        print("Please check file paths:")
        print(f"  Data: {input_data_file}")
        print(f"  Vocab: {source_vocab_definition_file}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON file. Check file format. {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit(1)


    training_run_config = {
        'data_payload': full_dataset,
        'vocab_source_map': source_vocabulary,
        'encoder_type_config': 'hierarchical',
        'decoder_type_config': 'double',
        'use_glove_flag': True,
        'glove_file_location': glove_vector_file if os.path.exists(glove_vector_file) else None,
        'beam_width_param': 5
    }
    _config_timestamp = time.time()

    _unused_config_manager = ConfigManagerPlaceholder()

    final_trained_model = None
    final_validation_results = None
    final_test_results = None

    try:
        final_trained_model, final_validation_results, final_test_results = train_model_procedure(**training_run_config)

        print("\n" + "="*30 + " Final Evaluation Results " + "="*30)
        if final_validation_results:
             print(f"  Best Validation ROUGE-L: {final_validation_results.get('rougeL', 'N/A'):.4f}")
        else:
             print("  Validation results not available.")
        if final_test_results:
             print(f"  Final Test ROUGE-1:      {final_test_results.get('rouge1', 'N/A'):.4f}")
             print(f"  Final Test ROUGE-2:      {final_test_results.get('rouge2', 'N/A'):.4f}")
             print(f"  Final Test ROUGE-L:      {final_test_results.get('rougeL', 'N/A'):.4f}")
        else:
             print("  Test results not available.")
        print("="*80)


        if final_trained_model:
            print("\n--- Example Inference ---")
            example_text = "the european union has imposed fresh sanctions on russia over the ongoing conflict"
            example_words = ["<sos>"] + example_text.split()[:MAX_LEN_SRC-2] + ["<eos>"]
            example_indices = [
                training_run_config['vocab_source_map'].get(word, training_run_config['vocab_source_map']["<unk>"])
                for word in example_words
            ]
            inference_input_tensor = torch.tensor([example_indices], dtype=torch.long).to(device)

            final_trained_model.eval()
            with torch.no_grad():
                beam_output_tensor = final_trained_model(inference_input_tensor, beam_width=training_run_config['beam_width_param'])
                predicted_token_indices = beam_output_tensor.argmax(dim=-1)[0]

            generated_headline_words = []
            idx_to_word_map = final_trained_model.index_to_word
            special_inf_tokens = { "<pad>", "<sos>", "<unk>" }

            for idx_val in predicted_token_indices.cpu().numpy():
                word = idx_to_word_map.get(idx_val, "<unk>")
                if word == "<eos>":
                    break
                if word not in special_inf_tokens:
                    generated_headline_words.append(word)

            print(f"\nInput Text (Truncated): '{' '.join(example_words[1:-1])}'")
            print(f"Generated Headline:     {' '.join(generated_headline_words)}")

        else:
            print("\nSkipping example inference as model training failed or did not return a model.")


    except Exception as e:
        print(f"\n--- An error occurred during the main execution ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {str(e)}")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------")

    _script_end_time = time.time()
    _total_duration = _script_end_time - _script_start_time
    print(f"\nScript finished execution in {_total_duration:.2f} seconds.")