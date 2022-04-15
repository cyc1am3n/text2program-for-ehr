# Base pkgs
import math
import os
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

# PyTorch pkgs
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

# Transformers pkgs
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput

from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers import BertConfig

# from transformers import (
#     BertModel,
#     BertOnlyMLMHead
# )

from .base_model import (
    BertModel,
    BertOnlyMLMHead,
    BertPreTrainedModel,
)

from utils.interpreter import MimicInterpreter
from utils.eval_utils import recover_pred_for_subwords, get_flag_for_execution_accuracy


logger = logging.get_logger(__name__)


class Text2TraceMaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.
    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[dict] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Text2TraceBertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, txt_len=100):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        
        self.tokenizer = tokenizer

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        
        self.config.tie_word_embeddings = True
        self.tie_weights()
        self.txt_len = txt_len

        cur_dir = os.getcwd()
        kg_path = f'{cur_dir}/data/db/mimicstar_kg/mimic_sparqlstar_kg.xml'
        ops_path  = f'{cur_dir}/data/db/mimicstar_kg/mimicprogram_operations.json'
        self.interpreter = MimicInterpreter(kg_path, ops_path)
        f = open(f'{cur_dir}/data/cond_look_up.json', encoding='UTF-8')
        self.tokenizer_look_up_json = eval(json.loads(f.read()))

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        answers=None,
        text_labels=None,
        trace_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        loss_dict = {
            'total_mlm_loss': None,
            'text_mlm_loss': None,
            'trace_mlm_loss': None,
            }
        masked_lm_loss = 0
        
        # compute total MLM loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            loss_dict['total_mlm_loss'] = masked_lm_loss.item()
        
        # compute MLM loss (text part)
        if text_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            text_masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size) , text_labels.view(-1))
            loss_dict['text_mlm_loss'] = text_masked_lm_loss.item()

        # compute MLM loss (trace part)
        if trace_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            trace_masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size) , trace_labels.view(-1))
            loss_dict['trace_mlm_loss'] = trace_masked_lm_loss.item()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss, loss_dict) + output) if masked_lm_loss is not None else output
        
        return Text2TraceMaskedLMOutput(
            loss=masked_lm_loss, # total_mlm_loss
            loss_dict=loss_dict,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    def decode_for_evaluation(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        text_labels=None,
        answers=None,
        trace_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bsz, seq_len = input_ids.shape
        half_block_size = int(seq_len/2)
        txt_len = self.txt_len
        
        # if text part is corrupted, 
        if text_labels is not None:
            input_ids = text_labels * text_labels.not_equal(-100) + torch.cat([input_ids[:, :txt_len] * text_labels[:, :txt_len].eq(-100), input_ids[:, txt_len:]], axis=1)
        
        curr_ids = input_ids[:, :txt_len]
        sos_ids = input_ids.new(bsz, 1).fill_(self.tokenizer.sep_token_id)
        mask_ids = input_ids.new(bsz, 1).fill_(self.tokenizer.mask_token_id)
        assert sos_ids.ndim == curr_ids.ndim == mask_ids.ndim
        
        curr_ids = torch.cat([curr_ids, sos_ids], axis=1)
        
        output_ids = []
        output_ids.append(curr_ids)
        
        next_pos = txt_len+1 # ???
        
        while next_pos < seq_len:
            
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)
            curr_length = list(curr_ids.size())[1]
            curr_attention_mask = attention_mask[:, :curr_length, :curr_length]
            curr_token_type_ids = token_type_ids[:, :curr_length]
            
            outputs = self.bert(
                curr_ids, # input_ids,
                curr_attention_mask, # attention_mask=attention_mask,
                curr_token_type_ids, # token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            sequence_output = outputs[0]
            # sequence_output = (bsz, seq_len, dim)
            last_hidden = sequence_output[:, -1, :]
            prediction_scores = self.cls(last_hidden)
            
            _, max_ids = torch.max(prediction_scores, dim=1)
            output_ids.append(max_ids.unsqueeze(1))
            
            # setup for next loop
            curr_ids[:, -1] = max_ids
            next_pos += 1
            
        output_ids = torch.cat(output_ids, dim=1)
        
        '''get metric (lf_acc)'''
        # we have to construct ground truth labels (due to fine-tune data_collator type)
        ground_labels = labels.not_equal(-100) * labels + (labels.eq(-100)) * input_ids

        ex_cnt = 0
        for b_idx in range(bsz):
            gt = ground_labels[b_idx][txt_len:]
            pred = output_ids[b_idx][txt_len:]
            answer = answers[b_idx]
                    
            # gt: cut by sep token
            gt_eos_idx = gt.not_equal(0).sum()
            gt = gt[:gt_eos_idx]
                
            # pred: cut by sep token if it exists
            if len(torch.nonzero(pred==self.tokenizer.sep_token_id)) > 1:
                pred_eos_idx = torch.nonzero(pred==self.tokenizer.sep_token_id)[1].item()
                pred = pred[:pred_eos_idx+1]
            else:
                pass
            
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)

            recover_pred = recover_pred_for_subwords(pred, self.tokenizer, self.tokenizer_look_up_json)
            ex_cnt += get_flag_for_execution_accuracy(recover_pred, self.interpreter, answer)

        result = ex_cnt
        return result
    
    
class Text2TraceBertForGeneration(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, tokenizer, txt_len=100):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
            
        self.tokenizer = tokenizer

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()
        self.txt_len = txt_len
        # self.tie_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        text_labels=None,
        trace_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bsz, seq_len = input_ids.shape
        half_block_size = int(seq_len/2)
        txt_len = self.txt_len

        curr_ids = input_ids[:, :txt_len]
        sos_ids = input_ids.new(bsz, 1).fill_(self.tokenizer.sep_token_id)
        mask_ids = input_ids.new(bsz, 1).fill_(self.tokenizer.mask_token_id)
        assert sos_ids.ndim == curr_ids.ndim == mask_ids.ndim
        
        curr_ids = torch.cat([curr_ids, sos_ids], axis=1)
        
        output_ids = []
        output_ids.append(curr_ids)
        
        next_pos = txt_len+1
        
        while next_pos < seq_len:
            
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)
            curr_length = list(curr_ids.size())[1]
            curr_attention_mask = attention_mask[:, :curr_length, :curr_length]
            curr_token_type_ids = token_type_ids[:, :curr_length]
            
            outputs = self.bert(
                curr_ids, # input_ids,
                curr_attention_mask, # attention_mask=attention_mask,
                curr_token_type_ids, # token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            sequence_output = outputs[0]
            # sequence_output = (bsz, seq_len, dim)
            last_hidden = sequence_output[:, -1, :]
            prediction_scores = self.cls(last_hidden)
            
            _, max_ids = torch.max(prediction_scores, dim=1)
            output_ids.append(max_ids.unsqueeze(1))
            
            # setup for next loop
            curr_ids[:, -1] = max_ids
            next_pos += 1 
            
        output_ids = torch.cat(output_ids, dim=1)        
        return output_ids
    
        
    def do_beam_search(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        answers=None,
        text_labels=None,
        trace_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        search_beam_size=3,
    ):  
        
        batch_size, seq_len = input_ids.shape
        half_block_size = int(seq_len/2)
        txt_len = self.txt_len
        vocab_size = len(self.tokenizer)


        # init settings
        curr_ids = input_ids[:, :txt_len] # (bsz, txt_len)
        sos_ids = input_ids.new(batch_size, 1).fill_(self.tokenizer.sep_token_id) # [SEP] = 102
        mask_ids = curr_ids.new(batch_size, 1).fill_(self.tokenizer.mask_token_id) # [MASK] = 103
        assert sos_ids.ndim == curr_ids.ndim == mask_ids.ndim
        
        curr_ids = torch.cat([curr_ids, sos_ids], axis=1) # starting point: TEXT(size: 128 or 256) + [SEP](size: 1)
        
        
        next_pos = txt_len+1
        K = search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        step_probs = []
        step_whole_probs = []
        step_entropy = []
        # partial_seqs = []
        # forbid_word_mask = None
        # buf_matrix = None

        while next_pos < seq_len:
            
            # construct current inputs
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)
            curr_length = list(curr_ids.size())[1]
            curr_attention_mask = attention_mask[:, :curr_length, :curr_length]
            curr_token_type_ids = token_type_ids[:, :curr_length]
            
            # output for current inputs
            outputs = self.bert(
                curr_ids,
                curr_attention_mask,
                curr_token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            # sequence_output = (bsz, seq_len, dim) -> (bsz * beam, seq_len, dim)
            sequence_output = outputs[0]
            last_hidden = sequence_output[:, -1, :] # (bsz, dim) -> (bsz * beam_size, dim)
            prediction_scores = self.cls(last_hidden) # (bsz, vocab_size) -> (bsz * beam_size, dim)
            prob = F.softmax(prediction_scores, dim=-1)
            log_scores = torch.log(prob)#F.log_softmax(prediction_scores, dim=-1) # log_scores: (batch_size, vocab_size) -> (bsz * beam, vocab)
            kk_entropy = torch.sum(prob * log_scores, dim=-1).unsqueeze(1) * (-1) # (bsz * beam)

            # choose top-k logits and indices
            ## first, (kk_scores, kk_ids) = (batch_size, beam_size)
            ## after that, (kk_scores, kk_ids) = (batch_size*beam_size, beam_size)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            kk_prob, _ = torch.topk(prob, k=K)#prob[:, kk_ids]

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:] # bsz, 1, seq_len
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x
            
            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:]) # (bsz, beam_size, curr_seq_len)
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y
            
            if len(total_scores) == 0: # for the first time,
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                k_prob = torch.reshape(kk_prob, [batch_size, K])
                #k_entropy = torch.reshape(kk_entropy, [batch_size, K])
                k_whole_prob = prob.repeat(1, K).reshape([batch_size, K, -1])
                k_entropy = kk_entropy.repeat(1, K)
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
                # print('start:', k_scores, k_ids)
            else:
                last_eos = torch.reshape(beam_masks[-1], [batch_size * K, 1])
                last_seq_scores = torch.reshape(total_scores[-1], [batch_size * K, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, _k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.floor_divide(_k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                kk_prob = torch.reshape(kk_prob, [batch_size, K * K])
                kk_whole_prob = prob.reshape([batch_size, K, -1])
                kk_whole_prob = kk_whole_prob.repeat_interleave(K, dim=1)
                kk_entropy = kk_entropy.repeat(1, K)
                kk_entropy = torch.reshape(kk_entropy, [batch_size, K * K])

                k_ids = torch.gather(kk_ids, 1, _k_ids)
                k_prob = torch.gather(kk_prob, 1, _k_ids)
                k_whole_prob = torch.gather(kk_whole_prob, 1, _k_ids.repeat_interleave(vocab_size).reshape([batch_size, K, -1]))
                k_entropy = torch.gather(kk_entropy, 1, _k_ids)
                
                curr_ids = select_beam_items(curr_ids, back_ptrs.long())
                
                
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            step_probs.append(k_prob)
            step_whole_probs.append(k_whole_prob)
            step_entropy.append(k_entropy)
            beam_masks.append(torch.eq(k_ids, self.tokenizer.sep_token_id).float())
            total_scores.append(k_scores)

            if next_pos == txt_len+1: # for the first time,
                curr_ids = first_expand(curr_ids)               # (bsz * beam_size, curr_seq_len)
                input_ids = first_expand(input_ids)             # (bsz * beam_size, seq_len)
                token_type_ids = first_expand(token_type_ids)   # (bsz * beam_size, seq_len)
                attention_mask = first_expand(attention_mask)   # (bsz * beam_size, seq_len, seq_len)
                mask_ids = first_expand(mask_ids)               # (bsz * beam_size, 1)
            
            # fill out the [MASK]'s position with stretched ids
            curr_ids[:, curr_length-1] = torch.reshape(k_ids, [batch_size * K])
            next_pos += 1
            
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_probs = [x.tolist() for x in step_probs]
        step_whole_probs = [x.tolist() for x in step_whole_probs]
        step_entropy = [x.tolist() for x in step_entropy]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq':[], 'scores': [], 'probs':[], 'whole_probs':[], 'entropy':[]}
        for b in range(batch_size):
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            wids_list_for_prob = [x[b] for x in step_probs]
            wids_list_for_whole_prob = [x[b] for x in step_whole_probs]
            wids_list_for_entropy = [x[b] for x in step_entropy]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.tokenizer.sep_token_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1
            
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.tokenizer.sep_token_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
                traces['probs'].append([0])
                traces['whole_probs'].append([0] * vocab_size)
                traces['entropy'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                seq_prob = [wids_list_for_prob[frame_id][pos_in_frame]]
                seq_whole_prob = [wids_list_for_whole_prob[frame_id][pos_in_frame]]
                seq_entropy = [wids_list_for_entropy[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                    seq_prob.append(wids_list_for_prob[fid - 1][pos_in_frame])
                    seq_whole_prob.append(wids_list_for_whole_prob[fid - 1][pos_in_frame])
                    seq_entropy.append(wids_list_for_entropy[fid - 1][pos_in_frame])
                seq.reverse()
                seq_prob.reverse()
                seq_whole_prob.reverse()
                seq_entropy.reverse()
                traces['pred_seq'].append(seq)
                traces['probs'].append(seq_prob)
                traces['whole_probs'].append(seq_whole_prob)
                traces['entropy'].append(seq_entropy)
        
        def _pad_sequence(sequences, txt_len, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, txt_len:txt_len+length, ...] = tensor
            return out_tensor

        for k in ('pred_seq', 'probs', 'whole_probs','entropy'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.int if k == 'pred_seq' else torch.float
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, txt_len, seq_len, padding_value=0).to(input_ids.device)

        output_ids = traces['pred_seq']
        output_prob = traces['probs'][:, txt_len:]
        output_entropy = traces['entropy'][:, txt_len:]
        output_whole_prob = traces['whole_probs'][:, txt_len:]

        return output_ids, output_prob, output_entropy, output_whole_prob

    def do_top_k_top_p_sampling(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        answers=None,
        text_labels=None,
        trace_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        do_sample=True,
        top_k=0,
        top_p=0.9,
        num_return_sequences=1,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bsz, seq_len = input_ids.shape
        txt_len = self.txt_len

        curr_ids = input_ids[:, :txt_len]                                                           # (bsz, txt_len)
        curr_ids = torch.repeat_interleave(curr_ids, num_return_sequences, 0)                       # (bsz * num_ret_seq, txt_len)
        sos_ids = input_ids.new(bsz * num_return_sequences, 1).fill_(self.tokenizer.sep_token_id)   # (bsz * num_ret_seq, 1)
        mask_ids = input_ids.new(bsz * num_return_sequences, 1).fill_(self.tokenizer.mask_token_id) # (bsz * num_ret_seq, 1)

        attention_mask = torch.repeat_interleave(attention_mask, num_return_sequences, 0)           # (bsz * num_ret_seq, seq_len, seq_len)
        #breakpoint()
        token_type_ids = torch.repeat_interleave(token_type_ids, num_return_sequences, 0)           # (bsz * num_ret_seq, seq_len)
        assert sos_ids.ndim == curr_ids.ndim == mask_ids.ndim
        
        curr_ids = torch.cat([curr_ids, sos_ids], axis=1)
        
        output_ids, output_prob, output_entropy = [], [], []
        output_ids.append(curr_ids)
        
        next_pos = txt_len + 1
        
        while next_pos < seq_len:
            
            curr_ids = torch.cat([curr_ids, mask_ids], axis=1)
            curr_length = list(curr_ids.size())[1]
            curr_attention_mask = attention_mask[:, :curr_length, :curr_length]
            curr_token_type_ids = token_type_ids[:, :curr_length]
            
            outputs = self.bert(
                curr_ids, # input_ids,
                curr_attention_mask, # attention_mask=attention_mask,
                curr_token_type_ids, # token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            sequence_output = outputs[0]                # (bsz * num_ret_seq, seq_len, dim)
            last_hidden = sequence_output[:, -1, :]     # (bsz * num_ret_seq, dim)
            prediction_scores = self.cls(last_hidden)   # (bsz * num_ret_seq, vocab)
            prob = F.softmax(prediction_scores, dim=1)
            log_prob = torch.log(prob)                                                          # (bsz * num_ret_seq, vocab)
            entropy = torch.sum(prob * log_prob, dim=1).unsqueeze(1) * (-1)                     # (bsz * num_ret_seq, 1)

            filtered_logits = self.fixed_top_k_top_p_filtering(prediction_scores, top_k, top_p) # (bsz * num_ret_seq, vocab)
            
            if do_sample:
                filtered_prob = F.softmax(filtered_logits, dim=1)
                next_token = torch.multinomial(filtered_prob, 1)
            else:
                _, next_token = torch.max(prob, 1)
                next_token = next_token.unsqueeze(1)
            output_ids.append(next_token)
            token_prob = torch.gather(prob, 1, next_token)  # (bsz * num_ret_seq, 1)
            output_prob.append(token_prob)
            output_entropy.append(entropy)
            
            
            # setup for next loop
            curr_ids[:, -1] = next_token.squeeze(1)
            next_pos += 1 
            
        output_ids = torch.cat(output_ids, dim=1)        
        output_prob = torch.cat(output_prob, dim=1)
        output_entropy = torch.cat(output_entropy, dim=1)
        return output_ids, output_prob, output_entropy

    def fixed_top_k_top_p_filtering(self, logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
            Args:
                logits: logits distribution shape (..., vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
        """
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs >= top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits