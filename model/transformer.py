# Base pkgs
import math
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
# PyTorch pkgs
import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F

# Transformers pkgs
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers import PretrainedConfig, AutoConfig

from typing import Optional

from utils.interpreter import MimicInterpreter
from utils.eval_utils import recover_pred_for_subwords, get_flag_for_execution_accuracy



logger = logging.get_logger(__name__)


class Text2TraceT5Model(PreTrainedModel):
    def __init__(
        self,
        model_name_or_path = None,
        config: Optional[PretrainedConfig] = None,
        tokenizer = None,
    ):
        if config is None:
            config = AutoConfig.from_pretrained('t5-base')
        sos_token_id = tokenizer.convert_tokens_to_ids('<s>')
        if tokenizer.decode(sos_token_id) == '<unk>':
            raise ValueError()
        config.decoder_start_token_id = sos_token_id
        self.config = config
        super().__init__(config)

        from transformers import T5ForConditionalGeneration
        if model_name_or_path is not None:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                config = config
            )
        else:
            raise ValueError()

        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer

        cur_dir = os.getcwd()
        kg_path = f'{cur_dir}/data/db/mimicstar_kg/mimic_sparqlstar_kg.xml'
        ops_path  = f'{cur_dir}/data/db/mimicstar_kg/mimicprogram_operations.json'
        self.interpreter = MimicInterpreter(kg_path, ops_path)
    
    def get_input_embeddings(self):
        return self.model.shared

    def set_input_embeddings(self, new_embeddings):
        self.model.shared = new_embeddings
        self.model.encoder.set_input_embeddings(new_embeddings)
        self.model.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.model.lm_head

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        decoder_head_mask=None,
        text_labels=None,
        answers=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            labels = torch.cat([labels[:, 1:], labels.new(labels.shape[0], 1).fill_(-100)], axis=1)
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            decoder_input_ids = decoder_input_ids,
            decoder_attention_mask = decoder_attention_mask,
            labels = labels,
            past_key_values = past_key_values,
            return_dict = return_dict
        )

        return outputs

    def decode_for_evaluation(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        text_labels=None,
        labels=None,
        answers=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bsz, seq_len = decoder_input_ids.shape
        #breakpoint()
        outputs = self.model.generate(
            input_ids=input_ids,
            decoder_start_token_id=self.config.decoder_start_token_id,
            attention_mask=attention_mask,
            max_length=seq_len
        )

        ex_cnt = 0
        for b_idx in range(bsz):
            # ignore CLS token
            gt = decoder_input_ids[b_idx][1:]
            pred = outputs[b_idx][1:]

            # cut by sep token
            gt_eos_idx = gt.not_equal(0).sum()
            gt = gt[:gt_eos_idx]

            # pred: cut by sep token if it exists
            if len(torch.nonzero(pred==self.tokenizer.eos_token_id)) > 0:
                pred_eos_idx = torch.nonzero(pred==self.tokenizer.eos_token_id)[0].item()
                pred = pred[:pred_eos_idx+1]
            else:
                pass
            # Pred might be not the same length of ground truth.
        
            answer = answers[b_idx]
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            ex_cnt += get_flag_for_execution_accuracy(pred, self.interpreter, answer)

        result = ex_cnt
        return result

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self.model._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past