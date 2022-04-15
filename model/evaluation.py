# Base pkgs
import logging
import re
import os
import json
import sys
import csv
sys.path.append('..')

import numpy as np
# from sumeval.metrics.rouge import RougeCalculator

import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import sqlite3
from sumeval.metrics.rouge import RougeCalculator

# Custom pkgs
from utils.interpreter import MimicInterpreter
from utils.eval_utils import is_digit, clean_text_for_spacing, clean_for_condition_quote, recover_condition_value


logger = logging.getLogger(__name__)


class EvalForMimicProgram:
    def __init__(self,
                 data_args,
                 training_args,
                 model_args,
                 tokenizer,
                 ):
        
        # arguments
        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        
        # tokenizer
        self.tokenizer = tokenizer
        
        self.test_trace_gt = self.load_question_ground_truth_data(self.data_args.test_data_file)

        # extra knowledge source
        cur_dir = os.getcwd()
        self.kg_path = f'{cur_dir}/data/db/mimicstar_kg/mimic_sparqlstar_kg.xml'
        self.ops_path  = f'{cur_dir}/data/db/mimicstar_kg/mimicprogram_operations.json'
        self.interpreter = MimicInterpreter(self.kg_path, self.ops_path)
        
        f = open(f'{cur_dir}/data/cond_look_up.json', encoding='UTF-8')
        self.tokenizer_look_up_json = eval(json.loads(f.read()))

        # For recover
        f = open(f'{cur_dir}/data/rel_obj_look_up.json', encoding='UTF-8')
        self.look_up = json.loads(f.read())
        
        
    def evaluate(self):
        return None
    
    def evaluate_step(self, model, batch, batch_idx):
        
        _keys_for_output = ['question', 'answer', 'pred', 'pred_tokens',
                            'sequence_score', 'sequence_entropy', 'ex_acc']
        if self.training_args.recover:
            _keys_for_output.extend(['recover_pred', 'recover_ex_acc'])
        results = {k: [] for k in _keys_for_output}
        tokenizer = self.tokenizer
        
        top_k = 50 if self.training_args.top_k is None else self.training_args.top_k
        top_p = 1.0 if self.training_args.top_p is None else self.training_args.top_p
        beam_size = self.training_args.beam_size
        num_samples = self.training_args.num_samples

        if self.model_args.encoder_decoder_type =='t5':
            bsz, seq_len = batch['decoder_input_ids'].shape
            seq_len = 350 if seq_len < 10 else seq_len
            do_sample = True if self.training_args.beam_size == 1 else False
            
            gen_model = model.model.model
            sos_token_id = tokenizer.convert_tokens_to_ids('<s>')
            if tokenizer.decode(sos_token_id) == '<unk>':
                raise ValueError()
        
            generation_output = gen_model.generate(
                input_ids=batch['input_ids'],
                decoder_start_token_id=sos_token_id,
                attention_mask=batch['attention_mask'],
                max_length=seq_len,
                do_sample=do_sample,
                num_beams=beam_size,
                num_return_sequences=num_samples,
                top_k=top_k,
                top_p=top_p,
                output_scores=True,
                return_dict_in_generate=True
                )
            
            outputs = generation_output['sequences']
            sequences_scores = generation_output['sequences_scores']                                            # (bsz * gen_size)
            logits = generation_output['scores']                                                                # (bsz * gen_size, seq_len, vocab)
            logits = torch.stack(logits, dim=1)[::int(beam_size/num_samples)].to(torch.float64)               # (bsz, seq_len - 1, vocab_size)
            output_prob = torch.softmax(logits, dim=2)                                                                 # (bsz, seq_len - 1, vocab_size)
            log_prob = torch.log_softmax(logits, dim=2)
            sequences_entropy = torch.sum(output_prob * log_prob, dim=2) * (-1)                                        # (bsz, seq_len - 1)
        
        elif self.model_args.encoder_decoder_type == 'unilm':
            bsz = batch['input_ids'].size(0)
            if self.training_args.beam_size == 1:
                do_sample = False if self.training_args.top_k is None and self.training_args.top_p is None else True
                outputs, output_prob, sequences_entropy = model.do_top_k_top_p_sampling(**batch, do_sample=do_sample,       # seq_len = inp_len + out_len
                                                                    top_k=top_k, top_p=top_p,                               # outputs:      (num_sample * batch_size, seq_len)
                                                                    num_return_sequences=self.training_args.num_samples)    # output_prob:  (num_sample * batch_size, out_len)
                
            elif self.training_args.beam_size > 1: # beam searching
                outputs, output_prob, sequences_entropy, output_whole_prob = model.do_beam_search(**batch, search_beam_size=self.training_args.beam_size)
        else: 
            raise ValueError("encoder_decoder_type is missing")
            
        questions, answers = self.test_trace_gt
         
        EVAL_BSZ = self.training_args.eval_batch_size
        
        for b_idx in range(bsz):
            question = questions[EVAL_BSZ*batch_idx+b_idx]
            answer = answers[EVAL_BSZ*batch_idx+b_idx]
            
            for sample_id in range(num_samples):
                if self.model_args.encoder_decoder_type == 't5':
                    end_token = tokenizer.eos_token_id
                    pred_tensor = outputs[b_idx * num_samples + sample_id][1:]
                    entropy = sequences_entropy[b_idx * num_samples + sample_id]
                    if len(torch.nonzero(pred_tensor==end_token)) > 0:
                        pred_eos_idx = torch.nonzero(pred_tensor==end_token)[0].item()
                        pred_tensor = pred_tensor[:pred_eos_idx+1]
                        entropy = entropy[:pred_eos_idx+1]
                
                elif self.model_args.encoder_decoder_type == 'unilm':
                    pred_tensor = outputs[b_idx * num_samples + sample_id][self.data_args.txt_len:]
                    entropy = sequences_entropy[b_idx * num_samples + sample_id]
                    if len(torch.nonzero(pred_tensor==tokenizer.sep_token_id)) > 1:
                        pred_eos_idx = torch.nonzero(pred_tensor==tokenizer.sep_token_id)[1].item()
                        pred_tensor = pred_tensor[:pred_eos_idx+1]
                        entropy = entropy[:pred_eos_idx]
                    elif len(torch.nonzero(pred_tensor==tokenizer.sep_token_id)) == 1:
                        pred_eos_idx = torch.nonzero(pred_tensor==tokenizer.sep_token_id)[0].item()
                        pred_tensor = pred_tensor[:pred_eos_idx+1]
                        entropy = entropy[:pred_eos_idx+1]

                else:
                    raise ValueError("encoder_decoder_type is missing")
                
                pred = tokenizer.decode(pred_tensor, skip_special_tokens=True)
                
                # post-preprocessing for Bert Tokenizer
                pred = pred.replace("don\'t", "do not")
                
                # pred -> pred tokens
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_tensor, skip_special_tokens=True)#tokenizer.tokenize(pred)
                
                # when we use bert tokenizer, some words like 'dob_year' will be truncated.
                # so we have to recover.
                if 'bert' in self.tokenizer.name_or_path:
                    pred = self._recover_pred_for_subwords(pred)
                if self.model_args.encoder_decoder_type =='t5':
                    pred = clean_text_for_spacing(pred)
                
                results["question"].append(question)
                results["answer"].append(answer)

                results["pred_tokens"].append(pred_tokens)
                
                results["pred"].append(pred)

                results["sequence_entropy"].append(entropy.tolist())
                
                ex_flag = self._get_flag_for_execution_accuracy(gt='unknown', pred=pred, interpreter=self.interpreter, answer=answer)
                results["ex_acc"].append(ex_flag)
                
                if self.training_args.recover:
                    pred_recover = recover_condition_value(pred, self.look_up)
                    results["recover_pred"].append(pred_recover)
                    
                    recover_ex_flag = self._get_flag_for_execution_accuracy(gt='unknown', pred=pred_recover, interpreter=self.interpreter, answer=answer)
                    results['recover_ex_acc'].append(recover_ex_flag)

        return results
        
    def load_question_ground_truth_data(self, data_file_path):
        with open(data_file_path) as json_file:
            questions, answers = [], []
            for line in json_file:
                dic = json.loads(line)
                questions.append(dic['question'])
                answers.append(dic['answer'])
        return questions, answers
    
    def _recover_pred_for_subwords(self, pred):
        new_pred = pred
        pattern_for_cond = re.compile(r"\'(.*?)\'")
        find_cond_pred = re.findall(pattern_for_cond, pred)
        find_cond_only_value = []
        for cond_val in find_cond_pred:
            if not(cond_val.startswith('/')) and not(cond_val in [str(i) for i in range(10)]):
                find_cond_only_value.append(cond_val)

        for cond_idx, cond_val in enumerate(find_cond_only_value):
            new_pred = new_pred.replace(cond_val, f'[COND]_{cond_idx}')
        new_pred = new_pred.replace(' ', '')

        for cond_idx, cond_val in enumerate(find_cond_only_value):
            cond_val = cond_val.strip()
            try:
                # recover condition value
                cond_val = self.tokenizer_look_up_json[' '.join(self.tokenizer.tokenize(cond_val))]
            except:
                pass
            new_pred = new_pred.replace(f'[COND]_{cond_idx}', cond_val)
        return new_pred
    
    def _get_flag_for_execution_accuracy(self, gt, pred, interpreter, answer):
        pred = clean_text_for_spacing(pred)
        pred = clean_for_condition_quote(pred)
        try:
            if gt != 'unknown':
                gt = clean_for_condition_quote(gt)
                output_gt = interpreter.execute_trace(gt)
                output_pred = interpreter.execute_trace(pred)
                if type(output_gt) not in [list, np.ndarray]:
                    output_gt = [output_gt]
                if type(output_pred) not in [list, np.ndarray]:
                    output_pred = [output_pred]
                if type(output_gt[0]) == np.ndarray:
                    result1 = sorted(output_gt[0].tolist()) == sorted(output_pred[0].tolist()) and sorted(output_gt[1].tolist()) == sorted(output_pred[1].tolist())
                    result2 = sorted(output_gt[0].tolist()) == sorted(output_pred[1].tolist()) and sorted(output_gt[1].tolist()) == sorted(output_pred[0].tolist())
                    return result1 or result2
                return sorted(output_gt) == sorted(output_pred)
            else:
                if type(answer) is str:
                    answer = [list(item) for item in set(tuple(row) for row in eval(answer))]
                else:
                    answer = [list(item) for item in set(tuple(row) for row in eval(str(answer)))]
            output_gt = answer
            output_pred = interpreter.execute_trace(pred)

            ex_flag = False
            if len(answer[0]) == 2:
                first, second = [], []
                for item in answer:
                    first.append(str(item[0]))
                    second.append(str(item[1]))
                
                if type(output_pred) == list and len(output_pred) == 2:
                    first, second = sorted(list(set(first))), sorted(list(set(second)))
                    first = [float(item) if is_digit(item) else item for item in first]
                    first = [item for item in first if item != 'None']
                    first = [str(item).replace('\\', ' ') if '\\' in str(item) else item for item in first]
                    second = [float(item) if is_digit(item) else item for item in second]
                    second = [item for item in second if item != 'None']
                    second = [str(item).replace('\\', ' ') if '\\' in str(item) else item for item in second]

                    pred_first, pred_second = sorted(list(set(output_pred[0]))), sorted(list(set(output_pred[1])))
                    pred_first = [float(item) if is_digit(item) else item for item in pred_first]
                    pred_first = [str(item).replace('\\', ' ') if '\\' in str(item) else item for item in pred_first]
                    pred_second = [float(item) if is_digit(item) else item for item in pred_second]
                    pred_second = [str(item).replace('\\', ' ') if '\\' in str(item) else item for item in pred_second]
                    
                    if (first == pred_first and second == pred_second) or (first == pred_second and second == pred_first):
                        ex_flag = True
            else:
                
                output_gt = [[str(item[0]).replace('\\', ' ')] if '\\' in str(item[0]) else item for item in answer]
                output_gt = sorted(output_gt)
                output_gt = [[float(item[0])] if is_digit(item[0]) else item for item in output_gt]
                
                output_pred = [[output_pred]] if type(output_pred) is float else output_pred.reshape((-1, 1)).tolist()
                output_pred = [list(item) for item in set(tuple(row) for row in output_pred)]
                
                output_pred = [[str(item[0]).replace('\\', ' ')] if '\\' in str(item[0]) else item for item in output_pred]
                output_pred = sorted(output_pred)
                output_pred = [[float(item[0])] if is_digit(item[0]) else item for item in output_pred]
                
                result = output_pred == output_gt

                if result:
                    ex_flag = True
                else:
                    ex_flag = False
        except:
            ex_flag = False
        return ex_flag