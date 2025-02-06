import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.output_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i: i+max_length]
            output_chunk = token_ids[i+1: i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.output_ids.append(torch.tensor(output_chunk))


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]


def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):

    tokenizer = tiktoken.get_encoding('gpt2')

    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            drop_last=drop_last, num_workers=num_workers)

    return dataloader
