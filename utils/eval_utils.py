import re
import csv
import pandas
import random
import json
import itertools
import numpy as np
from operator import itemgetter
from sumeval.metrics.rouge import RougeCalculator

from utils.interpreter import MimicInterpreter

# util function for evaluation
def gather_evaluation_outputs(epoch_outputs, recover=False):
    seq_entropy, ex_flag = [], []
    pred, pred_tokens = [], []
    questions, answers = [], []
    if recover:
        recover_ex_flag = []
        recover_pred = []

    for step_outputs in epoch_outputs:
        seq_entropy += step_outputs['sequence_entropy']
        ex_flag += step_outputs['ex_acc']
            
        pred += step_outputs['pred']
        pred_tokens += step_outputs['pred_tokens']
        questions += step_outputs['question']
        answers += step_outputs['answer']
        
        if recover:
            recover_ex_flag += step_outputs['recover_ex_acc']
            recover_pred += step_outputs['recover_pred']

    return_dict = {"ex_flag": ex_flag, "pred": pred, "pred_tokens": pred_tokens,
            "question":questions, "answer":answers, "sequence_entropy": seq_entropy}
    if recover:
        return_dict.update({"recover_ex_flag": recover_ex_flag, "recover_pred": recover_pred})
    return return_dict


# util function for save decode output file
def write_decode_output_file(save_file_path, save_file, recover=False):
    fieldnames = ["idx", "pred", "pred_tokens", "sequence_entropy",
                "ex_flag", "question", "answer"]
    if recover:
        fieldnames.extend(["recover_pred", "recover_ex_flag"])
    with open(save_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writerow({k:k for k in fieldnames}) # write the first column
        for idx in range(len(save_file['pred'])):
            row = {k:(save_file[k][idx] if k != "idx" else idx) for k in fieldnames}
            writer.writerow(row)
        csvfile.close()


def is_digit(str):
    try:
        tmp = float(str)
        return True
    except ValueError:
        return False

def recover_pred_for_subwords(pred, tokenizer, look_up):
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
                cond_val = look_up[' '.join(tokenizer.tokenize(cond_val))]
            except:
                pass
            new_pred = new_pred.replace(f'[COND]_{cond_idx}', cond_val)
        return new_pred

def clean_text_for_spacing(trace):
    # find condition vlaues in gen_entset_equal
    pattern_for_cond = re.compile(r"gen_entset_equal[\s]*\((.*?)\)<exe>")
    find_cond_trace = re.findall(pattern_for_cond, trace)
    # split condition into key, value
    p_for_split_key_value = re.compile(r"'(.*?)','(.*?)'")
    condition_values = []
    for i, cond in enumerate(find_cond_trace):
        filtered = re.findall(p_for_split_key_value, cond)
        if filtered:
            key, value = filtered[0]
            condition_values.append(value)
            trace = trace.replace(f"\'{value}\'", f"\'[COND]_{i}\'")
    # remove space in remained trace
    trace = trace.replace(' ', '')
    for i, cond in enumerate(condition_values):
        trace = trace.replace(f"\'[COND]_{i}\'", f"\'{cond}\'")
    return trace

def clean_for_condition_quote(pred):
    condition_quote_pattern = re.compile(r"gen_entset_equal\(\'/(.*?)\)<exe>")
    find_cond_pred = re.findall(condition_quote_pattern, pred)
    for filtered in find_cond_pred:
        try:
            rel, cond = filtered.split('\',\'')
        except:
            continue
        if cond.count('\'') > 1:
            new_value = rel + '\',"' + cond[:-1] + '\"'
            pred = pred.replace(filtered, new_value)
    return pred

def get_flag_for_execution_accuracy(pred, interpreter, answer):
    pred = clean_text_for_spacing(pred)
    pred = clean_for_condition_quote(pred)
    try:
        if type(answer) is str:
            answer = [list(item) for item in set(tuple(row) for row in eval(answer))]
        else:
            answer = [list(item) for item in set(tuple(row) for row in eval(str(answer)))]
        output_gt = answer
        output_pred = interpreter.execute_trace(pred)
        ex_flag = False
        first_all_digit, second_all_digit = True, True
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

def find_best_topk(input_, pool_, rouge_type='l', k=5, return_score=False):
    assert rouge_type in ['l', 1, 2]
    rouge = RougeCalculator(stopwords=False, lang="en")
    score_ = []
    for itm in pool_:
        input_ = input_.lower()
        itm = str(itm).lower()
        if rouge_type in [1, 2]:
            score_.append(rouge.rouge_n(summary=input_, references=itm, n=rouge_type))
        else:
            score_.append(rouge.rouge_l(summary=input_, references=itm))

    if np.sum(score_) == 0:
        score_ = []
        input2_ = ' '.join(list(input_)).lower()
        for itm in pool_:
            itm2 = ' '.join(list(str(itm))).lower()
            score_.append(rouge.rouge_n(summary=input2_, references=itm2, n=1))
    sorted_topk_index = np.argsort(score_)[::-1][:k]
    if return_score:
        sorted_topk_score = sorted(score_, reverse=True)[:k]
        return itemgetter(*list(sorted_topk_index))(pool_), sorted_topk_score
    else:
        return itemgetter(*list(sorted_topk_index))(pool_)
    
def recover_condition_value(trace, look_up):
    pred = trace
    # find condition vlaues in gen_entset_equal
    pattern_for_cond = re.compile(r"gen_entset_equal\((.*?)\)<exe>")
    find_cond_pred = re.findall(pattern_for_cond, pred)
    # split condition into key, value
    p_for_split_key_value = re.compile(r"'(.*?)','(.*?)'")
    for i, cond in enumerate(find_cond_pred):
        filtered = re.findall(p_for_split_key_value, cond)
        if filtered:
            key, value = filtered[0]
            if key in look_up:
                candidates = look_up[key]
                recovered = find_best_topk(value, candidates, k=1)
                pred = pred.replace(f"'{value}'", f"'{recovered}'")
    return pred