# Base pkgs
import os
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

# Transformers pkgs
from transformers import AutoTokenizer, AutoModel

# Custom pkgs
from .dataset import get_dataset
from .data_collator import DataCollatorForBertGeneration, DataCollatorForLanguageModeling, DataCollatorForUniLMGeneration


class Text2TraceDataModule(pl.LightningDataModule):
    def __init__(self, data_args, model_args, training_args):
        super().__init__()
        
        # Arguments
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        
        # Load Tokenizer
        self.tokenizer = self._load_tokneizer(model_args=self.model_args)        
        self.setup(stage='test')
    def setup(self, stage: Optional[str] = None):
        if (stage == "fit") or (stage == "test"):
            # Get dataset
            self.train_dataset, self.eval_dataset, self.test_dataset = get_dataset(
                data_args=self.data_args,
                tokenizer=self.tokenizer,
                evaluate=self.training_args.do_eval,
                prediction=self.training_args.do_predict,
                encoder_decoder_type=self.model_args.encoder_decoder_type,
            )
            
        # Define Data Collator
        if self.model_args.encoder_decoder_type == 't5':
            if self.training_args.train_setting == 'pretrain':
                raise NotImplementedError()
            elif self.training_args.train_setting in ['finetune', 'decode']:
                self.data_collator = DataCollatorForBertGeneration(tokenizer=self.tokenizer,
                                                                prediction=self.training_args.do_predict,
                                                                mlm=self.data_args.mlm,
                                                                attention_mask_type=self.training_args.attention_mask_type)
            else:
                raise ValueError()
        elif self.model_args.encoder_decoder_type == 'unilm':
            if self.training_args.train_setting == 'pretrain':
                self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                                    mlm=self.data_args.mlm,
                                                                    mlm_probability=self.data_args.mlm_probability,
                                                                    attention_mask_type=self.training_args.attention_mask_type,
                                                                    txt_len=self.data_args.txt_len,
                                                                    trace_len=self.data_args.trace_len)
            elif self.training_args.train_setting in ['finetune', 'decode']:
                self.data_collator = DataCollatorForUniLMGeneration(tokenizer=self.tokenizer,
                                                                    mlm=self.data_args.mlm,
                                                                    mlm_probability=self.data_args.mlm_probability,
                                                                    attention_mask_type=self.training_args.attention_mask_type,
                                                                    txt_len=self.data_args.txt_len,
                                                                    trace_len=self.data_args.trace_len,
                                                                    training_setting=self.training_args.train_setting)
        else:
            raise ValueError()
        
        self.eval_data_collator = None
    
    def prepare_data(self):
        self.model_args
        if 'bart' == self.model_args.encoder_decoder_type:
            pretrained_name = 'facebook/bart-base'#self.model_args.encoder_name_or_path
        elif 't5' == self.model_args.encoder_decoder_type:
            if 't5-base' in self.model_args.model_name_or_path:
                pretrained_name = 't5-base'
            elif 't5-small' in self.model_args.model_name_or_path:
                pretrained_name = 't5-small'
        else:
            pretrained_name = 'bert-base-uncased'
        AutoTokenizer.from_pretrained(pretrained_name)
        AutoModel.from_pretrained(pretrained_name)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_args.train_batch_size,
            sampler=RandomSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.training_args.eval_batch_size,
            sampler=SequentialSampler(self.eval_dataset),
            collate_fn=self.data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_args.eval_batch_size,
            collate_fn=self.eval_data_collator if self.eval_data_collator is not None else self.data_collator,
            drop_last=self.training_args.dataloader_drop_last,
            num_workers=self.training_args.dataloader_num_workers,
            pin_memory=self.training_args.dataloader_pin_memory,
        )
        
    def _load_tokneizer(self, model_args):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false' # for deepspeed error message
        new_words = []
        from utils.schema_mimic_trace import MAP_UNUSED_VOCAB_TO_GENERIC
        new_words = list(MAP_UNUSED_VOCAB_TO_GENERIC.values())
            
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name,
                never_split=new_words,
            )
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                #'bert-base-uncased',
                never_split=new_words,
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
        # T5 tokenizer doesn't contain sos token
        if tokenizer.name_or_path in ['t5-base', 't5-small']:
            new_words.append('<s>')
        if new_words:
            tokenizer.add_tokens(new_words)
        return tokenizer