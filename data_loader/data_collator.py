# import random
# import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import torch

# from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of Tensors.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])


# def tolist(x: Union[List[Any], torch.Tensor]):
#     return x.tolist() if isinstance(x, torch.Tensor) else x


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use masked language modeling. If set to :obj:`False`, the labels are the same as the
            inputs with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for
            non-masked tokens and the value to predict for the masked token.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when :obj:`mlm` is set to :obj:`True`.
    .. note::
        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    # TODO: poisitional encoding (make variable length)
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    attention_mask_type: str = 'bi'
    txt_len: int = 100
    trace_len: int = 350

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        # # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], (dict, BatchEncoding)):
        #     batch = self.tokenizer.pad(examples, return_tensors="pt")
        # else:
        #     batch = {"input_ids": _collate_batch(examples, self.tokenizer)}
        
        if isinstance(examples[0], dict):
            batch = {k: None for k in examples[0].keys()}
            for k in examples[0].keys():
                batch[k] = torch.stack([examples[idx][k] for idx in range(len(examples))])

        #block_size = len(batch["input_ids"][0])
        
        # shift the position for all input information
        # batch = self.shift_trace_position(input_batch=batch, block_size=block_size, remove_length_keys=False)
        batch = self.shift_trace_position(input_batch=batch, trace_len=self.trace_len, remove_length_keys=False)

        # extend attention mask for bar mask
        if self.attention_mask_type == 'bar':
            batch["attention_mask"] = self.make_bar_attention_mask(input_batch=batch, txt_len=self.txt_len) # block_size=block_size
        elif self.attention_mask_type == 'bi':
            pass
        else: 
            raise NotImplementedError()
        
        for k in ["text_token_length", "trace_token_length"]:
            del(batch[k])
        
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
            
            trace_indices = batch["token_type_ids"]
            batch["text_labels"] = (1-trace_indices) * batch["labels"] + trace_indices * -100
            batch["trace_labels"] = trace_indices * batch["labels"] + (1-trace_indices) * -100
        # else:
        #     labels = batch["input_ids"].clone()
        #     if self.tokenizer.pad_token_id is not None:
        #         labels[labels == self.tokenizer.pad_token_id] = -100
        #     batch["labels"] = labels
        
        return batch

    def shift_trace_position(self, input_batch, trace_len, remove_length_keys=True): #block_size,
        include_keys = ["input_ids", "token_type_ids", "attention_mask"]
        exclude_keys = ["text_token_length", "trace_token_length"]
        assert list(input_batch.keys()) == (include_keys + exclude_keys)
        for k in include_keys:
            for idx, data in enumerate(input_batch[k]):
                trace_s_idx = input_batch["text_token_length"][idx]
                trace_e_idx = trace_s_idx + trace_len#int(block_size/2)
                input_batch[k][idx] = torch.hstack([
                    input_batch[k][idx][:trace_s_idx],
                    input_batch[k][idx][trace_e_idx:],
                    input_batch[k][idx][trace_s_idx:trace_e_idx],
                ])
        if remove_length_keys:
            for k in exclude_keys:
                del(input_batch[k])
        return input_batch
                
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def make_bar_attention_mask(self,
                                input_batch: dict,
                                txt_len: int,
                                # block_size: int,
                                prob_for_bidirectional=0.0):
        assert 'text_token_length' in input_batch.keys()
        assert 'trace_token_length' in input_batch.keys()
        
        bsz, seq_len = input_batch["input_ids"].shape
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = torch.zeros(bsz, seq_len, seq_len, dtype=torch.long)
        _tril_matrix = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))

        # (bsz)
        data_prob = torch.bernoulli(torch.full((1, bsz), prob_for_bidirectional)).bool().view(-1)

        for b_idx in range(bsz):
            num_txt_tokens = input_batch["text_token_length"][b_idx] # including [CLS]
            num_trace_tokens = input_batch["trace_token_length"][b_idx] # including 2*[SEP]
            
            second_st = txt_len #int(block_size/2)
            second_end = txt_len + num_trace_tokens #int(block_size/2) + num_trace_tokens

            # bidirectional attention mask
            if data_prob[b_idx]:
                # txt -> txt, trace -> trace (follows bi-directional)
                attention_mask[b_idx, :num_txt_tokens, :num_txt_tokens].fill_(1) # txt->txt part
                attention_mask[b_idx, second_st:second_end, second_st:second_end].fill_(1) # trace->trace part
            
            # bidriectional auto-regressive mask    
            else:
                # txt,trace -> txt (follows bi-direcitonal)
                attention_mask[b_idx, :num_txt_tokens, :num_txt_tokens].fill_(1) # txt->txt part
                attention_mask[b_idx, second_st:second_end, :num_txt_tokens].fill_(1) # trace->txt part
                
                # txt -> trace (follows bi-directional)
                attention_mask[b_idx, :num_txt_tokens, second_st:second_end].fill_(1) # txt->trace part
                
                # trace -> trace (follows uni-direcitonal)
                attention_mask[b_idx, second_st:second_end, second_st:second_end].copy_(
                    _tril_matrix[:second_end-second_st, :second_end-second_st])

        return attention_mask


@dataclass
class DataCollatorForUniLMGeneration:

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    attention_mask_type: str = 's2s'
    txt_len: int = 100
    trace_len: int = 350
    training_setting: str = 'finetune'
    
    # TODO: poisitional encoding (make variable length)
    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        # # Handle dict or lists with proper padding and conversion to tensor.
        # if isinstance(examples[0], (dict, BatchEncoding)):
        #     batch = self.tokenizer.pad(examples, return_tensors="pt")
        # else:
        #     batch = {"input_ids": _collate_batch(examples, self.tokenizer)}
        
        if isinstance(examples[0], dict):
            batch = {k: None for k in examples[0].keys()}
            for k in examples[0].keys():
                if k == 'answers':
                    batch[k] = [examples[idx][k] for idx in range(len(examples))]
                else:
                    batch[k] = torch.stack([examples[idx][k] for idx in range(len(examples))])
        
        block_size = len(batch["input_ids"][0])
        txt_len = self.txt_len
        half_block_size = int(block_size/2)
                        
        # shift the position for all input information
        batch = self.shift_trace_position(input_batch=batch, trace_len=self.trace_len,remove_length_keys=False)

        # extend attention mask for seq2seq mask in unilm
        batch["attention_mask"] = self.make_unilm_attention_mask(input_batch=batch, txt_len=self.txt_len)
        
        # if trace label doesn't exist, modify token type ids
        if self.training_setting == 'decode':
            batch["token_type_ids"][:, self.txt_len+1:].fill_(1)
        
        # # If special token mask has been preprocessed, pop it from the dict.
        # special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        special_tokens_mask = []
        for idx, data in enumerate(batch["input_ids"].tolist()):
            mask = self.tokenizer.get_special_tokens_mask(data, already_has_special_tokens=True)
            last_sep_token_idx = txt_len + batch["trace_token_length"][idx] - 1
            mask[last_sep_token_idx] = 0 # when applying s2s finetuning, we must include last <eos> tokens in trace part(In here, [SEP] does)
            special_tokens_mask.append(mask)
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.long)
                
        for k in ["text_token_length", "trace_token_length"]:
            del(batch[k])
        
        if self.mlm:
            if self.attention_mask_type == 's2s':
                
                # when masking, we ignore all tokens in the text part
                special_tokens_mask[:, :txt_len] = 1
                
                batch["input_ids"], batch["labels"] = self.mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
                
                # trace_indices = batch["token_type_ids"]
                batch["text_labels"] = None
                batch["trace_labels"] = batch["labels"].clone()
                
                
            elif self.attention_mask_type in [f's2s_mask_nlq_{num:02d}' for num in range(100)]: # Also, we're going to mask the text part!
                
                # take each mlm proability 
                text_mlm_probability = int(self.attention_mask_type.split('_')[-1]) * 0.01
                trace_mlm_probability = self.mlm_probability
                
                # masking nlq part (should be cloned)
                self.mlm_probability = text_mlm_probability
                text_input_ids, text_labels = self.mask_tokens(
                    batch["input_ids"][:, :txt_len].clone(), special_tokens_mask=None
                )
                
                special_tokens_mask[:, :txt_len] = 1
                
                # masking trace part (should be cloned)
                self.mlm_probability = trace_mlm_probability
                batch["input_ids"], batch["labels"] = self.mask_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
                
                batch["input_ids"][:, :txt_len] = text_input_ids
                batch["labels"][:, :txt_len] = text_labels
                
                # # concat each outputs (masked input_ids, corresponding labels)
                # batch["input_ids"] = torch.cat([text_input_ids, trace_input_ids], axis=1)
                # batch["labels"] = torch.cat([text_labels, trace_labels], axis=1)
                
                # separate labels into text and trace part
                trace_indices = batch["token_type_ids"].clone()
                batch["text_labels"] = (1-trace_indices) * batch["labels"].clone() + trace_indices * -100
                batch["trace_labels"] = trace_indices * batch["labels"].clone() + (1-trace_indices) * -100
                
            elif self.attention_mask_type in [f's2s_only_mask_nlq_{num:02d}' for num in range(100)]: 
                
                # take each mlm proability 
                text_mlm_probability = int(self.attention_mask_type.split('_')[-1]) * 0.01
                trace_mlm_probability = self.mlm_probability
                
                # masking nlq part (should be cloned)
                self.mlm_probability = text_mlm_probability
                text_input_ids, text_labels = self.mask_tokens(
                    batch["input_ids"][:, :txt_len].clone(), special_tokens_mask=None
                )
                
                special_tokens_mask[:, :txt_len] = 1
                
                # masking trace part (should be cloned)
                self.mlm_probability = trace_mlm_probability
                batch["input_ids"], batch["labels"] = self.mask_only_tokens(
                    batch["input_ids"], special_tokens_mask=special_tokens_mask
                )
                
                batch["input_ids"][:, :txt_len] = text_input_ids
                batch["labels"][:, :txt_len] = text_labels
                
                # # concat each outputs (masked input_ids, corresponding labels)
                # batch["input_ids"] = torch.cat([text_input_ids, trace_input_ids], axis=1)
                # batch["labels"] = torch.cat([text_labels, trace_labels], axis=1)
                
                # separate labels into text and trace part
                trace_indices = batch["token_type_ids"].clone()
                batch["text_labels"] = (1-trace_indices) * batch["labels"].clone() + trace_indices * -100
                batch["trace_labels"] = trace_indices * batch["labels"].clone() + (1-trace_indices) * -100
                
            else:
                raise ValueError("When fine-tuning, you should have to use s2s mask")
            
        else:
            # labels = batch["input_ids"].clone()
            # if self.tokenizer.pad_token_id is not None:
            #     labels[labels == self.tokenizer.pad_token_id] = -100
            # batch["labels"] = labels
            
            batch["labels"] = None
            batch["text_labels"] = None
            batch["trace_labels"] = None
            
        return batch


    def shift_trace_position(self, input_batch, trace_len, remove_length_keys=True):
        include_keys = ["input_ids", "token_type_ids", "attention_mask"]
        exclude_keys = ["text_token_length", "trace_token_length"]
        #assert list(input_batch.keys()) == (include_keys + exclude_keys)
        for k in include_keys:
            for idx, data in enumerate(input_batch[k]):
                trace_s_idx = input_batch["text_token_length"][idx]
                trace_e_idx = trace_s_idx + trace_len #int(block_size/2)
                input_batch[k][idx] = torch.hstack([
                    input_batch[k][idx][:trace_s_idx],
                    input_batch[k][idx][trace_e_idx:],
                    input_batch[k][idx][trace_s_idx:trace_e_idx],
                ])
        if remove_length_keys:
            for k in exclude_keys:
                del(input_batch[k])
        return input_batch
                
                
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    

    def mask_only_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

        # # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    
    def make_unilm_attention_mask(self,
                                  input_batch: dict,
                                  txt_len: int,
                                  prob_for_bidirectional=0.0):
        '''
        reference: https://github.com/LuoweiZhou/VLP/blob/765ce07a1dfd327a7fd3f484d4e19398e8c77121/vlp/seq2seq_loader.py#L362
        '''
        assert 'text_token_length' in input_batch.keys()
        assert 'trace_token_length' in input_batch.keys()
        
        bsz, seq_len = input_batch["input_ids"].shape
        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = torch.zeros(bsz, seq_len, seq_len, dtype=torch.long)
        _tril_matrix = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))

        # (bsz)
        data_prob = torch.bernoulli(torch.full((1, bsz), prob_for_bidirectional)).bool().view(-1)

        for b_idx in range(bsz):
            num_txt_tokens = input_batch["text_token_length"][b_idx] # including [CLS]
            num_trace_tokens = input_batch["trace_token_length"][b_idx] # including 2*[SEP]
            
            second_st = txt_len #int(block_size/2)
            second_end = txt_len + num_trace_tokens #int(block_size/2) + num_trace_tokens

            # bidirectional attention mask
            if data_prob[b_idx]:
                # txt -> txt, trace -> trace (follows bi-directional)
                attention_mask[b_idx, :num_txt_tokens, :num_txt_tokens].fill_(1) # txt->txt part
                attention_mask[b_idx, second_st:second_end, second_st:second_end].fill_(1) # trace->trace part
                
            else:
                # txt,trace -> txt (follows bi-direcitonal)
                attention_mask[b_idx, :num_txt_tokens, :num_txt_tokens].fill_(1) # txt->txt part
                if self.training_setting == "finetune":
                    attention_mask[b_idx, second_st:second_end, :num_txt_tokens].fill_(1) # trace->txt part
                elif self.training_setting == "decode":
                    attention_mask[b_idx, second_st:, :num_txt_tokens].fill_(1) # trace->txt part
                else:
                    raise ValueError()
                
                # trace -> trace (follows uni-direcitonal)
                if self.training_setting == "finetune":
                    attention_mask[b_idx, second_st:second_end, second_st:second_end].copy_(
                        _tril_matrix[:second_end-second_st, :second_end-second_st])
                elif self.training_setting == "decode":
                    # print(attention_mask[b_idx, second_st:, second_st:].shape)
                    # print(_tril_matrix[-second_st:, -second_st:].shape)
                    attention_mask[b_idx, second_st:, second_st:].copy_(
                        _tril_matrix[-self.trace_len:, -self.trace_len:]
                    )

        return attention_mask




@dataclass
class DataCollatorForBertGeneration:
    
    tokenizer: PreTrainedTokenizerBase
    prediction: bool = False
    mlm: bool = True
    attention_mask_type: str = 's2s'

    def __call__(self, examples: List[InputDataClass]) -> Dict[str, torch.Tensor]:
        assert type(examples[0]) == dict
        enc_features = [{'input_ids': e['encoder_input_ids']} for e in examples]
        dec_features = [{'input_ids': e['decoder_input_ids']} for e in examples]
        
        batch = self.tokenizer.pad(
            enc_features,
            padding="longest",
            max_length=512,
            pad_to_multiple_of=8, # https://discuss.huggingface.co/t/whats-a-good-value-for-pad-to-multiple-of/1481/2
            return_tensors="pt",
        )

        dec_batch =  self.tokenizer.pad(
            dec_features,
            padding="longest",
            max_length=512,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        
        # Append prefix `decoder_`
        for k in batch.copy():
            batch[f"decoder_{k}"] = dec_batch[k]

        # Label for generation
        batch['labels'] = batch["decoder_input_ids"].clone().detach()
        pad_indices = batch['labels'].eq(self.tokenizer.pad_token_id)
        batch['labels'][pad_indices] = -100

        # Causal attention mask
        n_ctx = dec_batch["input_ids"].size(-1)
        batch['decoder_attention_mask'] = torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, n_ctx, n_ctx)
        
        batch['text_labels'] = None
        if self.mlm and self.attention_mask_type in [f's2s_mask_nlq_{num:02d}' for num in range(100)]:
            text_mlm_probability = int(self.attention_mask_type.split('_')[-1]) * 0.01
            batch['input_ids'], batch['text_labels'] = self.mask_tokens(
                    batch["input_ids"].clone(), special_tokens_mask=None, mlm_probability=text_mlm_probability
                )
        if 'answers' in examples[0]:
            batch['answers'] = [e['answers'] for e in examples]
        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None, mlm_probability: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels