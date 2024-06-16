import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load the enwik8 dataset
    dataset = load_dataset("enwik8", num_proc=num_proc, split='train')

    # enwik8 is a single dataset with 100M characters. We need to split it: 90M for train, 5M for dev, and 5M for test.
    # train_test_split = dataset.train_test_split(test_size=10_000_000, seed=42)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

    # Organize splits in a dictionary for convenience
    split_dataset = {
        'train': train_test_split['train'],
        'val': test_val_split['test'],
        'test': test_val_split['train']
    }

    # Define the encoding function
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = {
        split: dset.map(process, remove_columns=['text'], desc=f"tokenizing {split}", num_proc=num_proc)
        for split, dset in split_dataset.items()
    }

    # Concatenate all the ids in each dataset into one large file for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()