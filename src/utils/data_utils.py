import os
import re
import json
from datasets import load_dataset

def get_raw_dataset(dataset_id, dataset_version, max_lines=None):
    cache_file = f"data/raw/ms_marco_{'all' if max_lines is None else max_lines}.txt"  # Path to the cached dataset file

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Downloading and saving dataset to {cache_file}...")
        dataset = load_dataset(dataset_id, dataset_version, split='train')
        with open(cache_file, "w", encoding='utf-8') as f:
            for item in dataset:
                json.dump(item, f)
                f.write("\n")
    else:
        print(f"Cache file found at {cache_file}. Using cached dataset.")

    # Proceed with loading the dataset from the cache file or using it directly
    with open(cache_file, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Cap the number of lines if max_lines is specified
    if max_lines is not None:
        dataset = dataset[:max_lines]

    return dataset

def get_clean_dataset(dataset):
    # Embed the number of lines in the file name
    num_lines = len(dataset)
    cache_file = f"data/processed/ms_marco_clean_{num_lines}_lines.json"

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Cleaning dataset and saving to {cache_file}...")
        cleaned_data = []
        for item in dataset:
            # Clean the dataset by removing unnecessary fields
            
            cleaned_item = {
                "query": re.sub('\W+', ' ', item["query"]).lower(),
                "passages": [re.sub('\W+', ' ', passage).lower() for passage in item["passages"]["passage_text"]],
            }
            cleaned_data.append(cleaned_item)
        
        with open(cache_file, "w") as f:
            json.dump(cleaned_data, f, indent=4)
    else:
        print(f"Cache file found at {cache_file}. Using cached cleaned dataset.")

    with open(cache_file, "r") as f:
        cleaned_dataset = json.load(f)

    return cleaned_dataset

def get_tokenised_dataset(clean_dataset, min_frequency=0):
    # Check if we already have a tokenized dataset with the same number of lines
    num_lines = len(clean_dataset)
    
    cache_file = f"data/processed/ms_marco_tkn_word_to_ids_{num_lines}_lines_minfreq_{min_frequency}.json"
    reverse_cache_file = f"data/processed/ms_marco_tkn_ids_to_words_{num_lines}_lines_minfreq_{min_frequency}.json"
    
    if os.path.exists(cache_file):
        print(f"Cache file found at {cache_file}. Using cached tokenized dataset.")
        with open(cache_file, "r") as f:
            result = json.load(f)
        
        with open(reverse_cache_file, "r") as f:
            reverse_result = json.load(f)
            reverse_result = {int(k): v for k, v in reverse_result.items()}

        return result, reverse_result

    print(f"Cache file not found. Tokenizing dataset and saving to {cache_file}...")
    
    word_frequency = {}

    for item in clean_dataset:
        # Tokenize the query and passages (replace with actual tokenization logic)
        tokenized_query = item["query"].lower().split()  # Replace with actual tokenization
        tokenized_passages = [passage.lower().split() for passage in item["passages"]]  # Replace with actual tokenization

        # Update word frequency
        for word in tokenized_query:
            word_frequency[word] = word_frequency.get(word, 0) + 1
        for passage in tokenized_passages:
            for word in passage:
                word_frequency[word] = word_frequency.get(word, 0) + 1

    # throw error if min_frequency is less than 0
    if min_frequency < 0:
        raise ValueError("min_frequency must be greater than or equal to 0")

    # Filter words based on min_frequency
    if min_frequency > 0:
        filtered_word_frequency = {word: freq for word, freq in word_frequency.items() if freq >= min_frequency}
        remaining_words = {word: idx for idx, word in enumerate(filtered_word_frequency.keys())}
        removed_words_count = len(word_frequency) - len(filtered_word_frequency)

        print(f"Tokenized dataset saved to {cache_file}.")

        print(f"Number of words in the original dataset: {len(word_frequency)}")
        print(f"Number of remaining words: {len(filtered_word_frequency)}")
        print(f"Number of removed words: {removed_words_count}")
        print(f"% of words removed: {removed_words_count / len(word_frequency) * 100:.2f}%")

        result = remaining_words
    else:
        words = {word: idx for idx, word in enumerate(word_frequency.keys())}

        print(f"Number of remaining words: {len(words)}")
        print(f"Number of removed words: 0")
    
        result = words

    # Add special tokens
    result["<UNK>"] = len(result)

    # Save the tokenized dataset to a cache file
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=4)

        print(f"Tokenized dataset saved to {cache_file}.")

    # Save the reverse of result to a separate file
    reverse_result = {int(idx): word for word, idx in result.items()}
    
    with open(reverse_cache_file, "w") as f:
        json.dump(reverse_result, f, indent=4)

    print(f"Reverse tokenized dataset saved to {reverse_cache_file}.")

    return result, reverse_result

def get_corpus(clean_dataset, vocab_to_int, min_frequency):
    # Embed the number of lines in the file name
    num_lines = len(clean_dataset)
    cache_file = f"data/processed/ms_marco_corpus_{num_lines}_lines_minfreq_{min_frequency}.json"

    if not os.path.exists(cache_file):
        print(f"Cache file not found. Creating corpus and saving to {cache_file}...")
        corpus = []
        # Save each query + passages in one sentence
        for item in clean_dataset:
            # Join the query and passages into a single string
            corpus_item = f"{item['query']} {' '.join(item['passages'])}"
            # Tokenize and replace words not in vocab_to_int with <UNK>
            corpus.append(corpus_item.split())

        # join all the sentences into one list
        corpus = [
            "<UNK>" if word not in vocab_to_int else word
            for sentence in corpus
            for word in sentence
        ]

        with open(cache_file, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print(f"Cache file found at {cache_file}. Using cached corpus.")

    with open(cache_file, "r") as f:
        corpus = json.load(f)

    return corpus