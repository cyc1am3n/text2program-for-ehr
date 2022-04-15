# Base pkgs
import json
import os
import pickle
import random
import time
import warnings
from typing import Dict, List, Optional
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

import logging
import dataclasses
from dataclasses import dataclass

# Transformers pkgs
from transformers.tokenization_utils import PreTrainedTokenizer

# Custom pkgs
from utils.data_args import DataTrainingArguments
from utils.data_utils import extract_semantic_from_template


logger = logging.getLogger(__name__)


class TextAndTraceDataset(Dataset):
    '''
        Use For UnifiedLM Model
        input_ids: [CLS] text_tokens [SEP] trace_tokens [SEP]
    '''
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, txt_len:int, trace_len:int): #block_size: int
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            # lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
            lines, answers = [], []
            err_cnt = 0
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    data = json.loads(line)
                    text, trace, answer = data['question'], data["trace"], data["answer"]
                    answers.append(answer)
                    lines.append([text, trace])
            
        batch_encoding = tokenizer(lines, add_special_tokens=True, padding='max_length', truncation=True, max_length=txt_len+trace_len) # block_size
        assert list(batch_encoding.keys()) == ['input_ids', 'token_type_ids', 'attention_mask']
        
        batch_encoding['text_token_length'], batch_encoding['trace_token_length'] = [], []
        for idx, input_ids in enumerate(batch_encoding['input_ids']):
            text_token_length, trace_token_length = self.get_paired_length(input_ids)
            batch_encoding['text_token_length'].append(text_token_length)
            batch_encoding['trace_token_length'].append(trace_token_length)
        
        # dict(list) -> list(dict)
        self.examples = []
        for idx in tqdm(range(len(batch_encoding['input_ids']))):
            example = {k: torch.tensor(batch_encoding[k][idx], dtype=torch.long) for k in batch_encoding.keys()}
            if answers:
                example['answers'] = answers[idx]
            self.examples.append(example)
            
        assert self.__len__() == len(lines)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    
    def get_paired_length(self, input_ids):
        # tokenizer.sep_token_id = 102
        if isinstance(input_ids, torch.Tensor):
            text_length = torch.nonzero((input_ids == 102))[0].item() # sep_token_pos1
            full_length = torch.nonzero((input_ids == 102))[1].item() # sep_token_pos2
            trace_length = full_length - text_length + 1
        elif isinstance(input_ids, list):
            text_length = input_ids.index(102) # sep_token_pos1
            full_length = input_ids.index(0) # sep_token_pos2
            trace_length = full_length - text_length
        return (text_length, trace_length)


class TextToTraceDataset(Dataset):
    '''
        Use For Encoder-Decoder model (transformer)
        input_ids: txt_tokens
        decoder_input_ids: [CLS] + trace_tokens + [SEP]
    '''
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info("Creating features from dataset file at %s", file_path)
        # load dataset
        err_cnt = 0
        with open(file_path, encoding="utf-8") as f:
            txt_lines, trace_lines = [], []
            answers = []
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    data = json.loads(line)        
                    text, trace, answer = data['question'], data["trace"], data['answer']
                    answers.append(answer)
                    txt_lines.append(text)
                    if tokenizer.name_or_path in ['t5-base', 't5-small']:
                        trace = '<s>' + trace
                    trace_lines.append(trace)
                    
        
        # make dict(list)
        batch_encoding = {'encoder_input': None, 'decoder_input': None}
        batch_encoding['encoder_input'] = tokenizer(txt_lines,
                                                add_special_tokens=False, 
                                                return_attention_mask=True,
                                                return_token_type_ids=False,
                                                max_length=block_size)
        batch_encoding['decoder_input'] = tokenizer(trace_lines,
                                                        add_special_tokens=True,
                                                        return_attention_mask=False,
                                                        return_token_type_ids=False,
                                                        max_length=block_size)
        
        # convert dict(list) to list(dict)
        self.examples = []
        for idx in tqdm(range(len(batch_encoding['encoder_input']['input_ids']))):
            example_encoder = {f'encoder_{k}': torch.tensor(batch_encoding['encoder_input'][k][idx], dtype=torch.long) for k in batch_encoding['encoder_input'].keys()}
            example_decoder = {f'decoder_{k}': torch.tensor(batch_encoding['decoder_input'][k][idx], dtype=torch.long) for k in batch_encoding['decoder_input'].keys()}
            if answers:
                example_answer = {'answers': answers[idx]}
                self.examples.append({**example_encoder, **example_decoder, **example_answer})
            else:
                self.examples.append({**example_encoder, **example_decoder})
        assert self.__len__() == len(txt_lines)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
    

def get_dataset(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    prediction: bool = False,
    encoder_decoder_type: str = 'unilm'
):
    def _dataset(file_path, block_size, encoder_decoder_type):
        if encoder_decoder_type == 'unilm':    
            return TextAndTraceDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                txt_len=data_args.txt_len,
                trace_len=data_args.trace_len,
                #block_size=block_size
            )
        elif encoder_decoder_type == 't5':
            return TextToTraceDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=block_size,
            )
    return (
        _dataset(data_args.train_data_file, data_args.block_size, encoder_decoder_type) if not(prediction) else None,
        _dataset(data_args.eval_data_file, data_args.block_size, encoder_decoder_type) if evaluate else None,
        _dataset(data_args.test_data_file, data_args.block_size, encoder_decoder_type) if prediction else None,
    )