import os
import glob
import json
import re
import time
from rdflib import Graph, URIRef
import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain


class MimicInterpreter:
    def __init__(self, kg_path, ops_path):
        self.kg = Graph()
        self.kg.parse(kg_path, format='xml', publicID='/')
        self.triples = self.kg2triples(self.kg)
        self.triples_num_idx = np.vectorize(self.isfloat)(self.triples['obj_str'])

        self.ops_path = ops_path
        self.idx2op, self.op2idx, self.idx2type, self.type2idx, self.op2argtypes_mat, self.op2outtype_mat, \
            self.max_args_over_ops = self.build_ops()

        self.n_ops = len(self.idx2op)
        self.n_types = len(self.idx2type)

    def build_ops(self):
        None_op = 'no_op'
        None_type = 'None'
        ops = []
        ops_c = Counter()
        types_c = Counter()
        max_args = 0
        with open(self.ops_path) as json_file:
            for line in json_file:
                ops.append(json.loads(line))

                n_args4op = len(ops[-1]['arg_types'])
                if n_args4op > max_args:
                    max_args = n_args4op

                ops_c.update([ops[-1]['name']])
                types_c.update(ops[-1]['arg_types'])
                types_c.update([ops[-1]['out_type']])

        idx2op = [None_op] + [op for op, count in ops_c.items()]
        idx2type = [None_type] + [types for types, count in types_c.items() if types != "None"]
        op2idx = {op: idx for idx, op in enumerate(idx2op)}
        type2idx = {_type: idx for idx, _type in enumerate(idx2type)}

        op2argtypes_mat = np.zeros(shape=(len(idx2op), max_args), dtype=np.long)
        op2outtype_mat = np.zeros(shape=(len(idx2op),), dtype=np.long)

        for op in ops:
            op2argtypes_mat[op2idx[op['name']], :] = [type2idx[_type] for _type in op['arg_types']]
            op2outtype_mat[op2idx[op['name']]] = type2idx[op['out_type']]

        return idx2op, op2idx, idx2type, type2idx, op2argtypes_mat, op2outtype_mat, max_args

    def kg2triples(self, kg):
        triples = dict()
        sub, rel, obj = [], [], []
        for t in kg:
            sub.append(t[0].toPython())
            rel.append(t[1].toPython())
            obj.append(str(t[2].toPython()))#.replace(' ', '')) # if you recover space for subword, do not remove space for obj

        self.sub_obj2id = self.build_vocab(sub + obj)
        self.rel2id = self.build_vocab(rel)

        self.id2sub_obj = {i: key for key, i in self.sub_obj2id.items()}
        self.id2rel = {i: key for key, i in self.rel2id.items()}

        self.np_sub_obj2id = np.vectorize(lambda x: self.sub_obj2id[x])
        self.np_rel2id = np.vectorize(lambda x: self.rel2id[x])

        self.np_id2sub_obj = np.vectorize(lambda x: self.id2sub_obj[x])
        self.np_id2rel = np.vectorize(lambda x: self.id2rel[x])

        triples['sub'] = self.np_sub_obj2id(np.array(sub))
        triples['rel'] = self.np_rel2id(np.array(rel))
        triples['obj'] = self.np_sub_obj2id(np.array(obj))
        triples['obj_str'] = np.array(obj)

        return triples

    def build_vocab(self, data, min_freq=1):
        PAD_TOKEN = '<PAD>'
        PAD_TOKEN_IDX = 0
        UNK_TOKEN = '<UNK>'
        UNK_TOKEN_IDX = 1

        SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN]

        idx2word = SPECIAL_TOKENS + [word for word, count in Counter(data).items() if count >= min_freq]
        word2idx = {word: idx for idx, word in enumerate(idx2word)}

        return word2idx

    def isfloat(self, val):
        if type(val) != np.ndarray:
            val = np.array(val)
        try:
            val.astype(float)
            return True
        except:
            return False

    def obj_to_nl(self, rel_obj):
        p = re.compile('/[a-zA-Z0-9_]*/[a-zA-Z0-9_]*')
        if p.match(rel_obj):
            rel_obj = rel_obj.split('/')[-1]
        return rel_obj

    # hopping(down): entSet rel > entSet
    def gen_entSet_down(self, entSet, rel_ent, readable=True):
        if entSet is None:
            return None

        if type(entSet) != np.ndarray:
            entSet = np.array([entSet])

        if len(entSet) == 0:
            return np.array([], dtype=np.int)

        if readable:
            entSet = self.np_sub_obj2id(entSet)
            rel_ent = self.np_rel2id(rel_ent)

        triple_idx = np.where(np.in1d(self.triples['sub'], entSet))[0]
        rels = self.triples['rel'][triple_idx]
        rel_idx = np.where(rels == rel_ent)[0]
        obj_set = self.triples['obj'][triple_idx[rel_idx]]

        result = self.np_id2sub_obj(obj_set) if readable else obj_set
        return result

    # hopping(up): entSet < rel entSet
    def gen_entSet_up(self, rel_ent, entSet, readable=True):
        if entSet is None:
            return None

        if type(entSet) != np.ndarray:
            entSet = np.array([entSet])

        if len(entSet) == 0:
            return np.array([], dtype=np.int)

        if readable:
            entSet = self.np_sub_obj2id(entSet)
            rel_ent = self.np_rel2id(rel_ent)

        triple_idx = np.where(np.in1d(self.triples['obj'], entSet))[0]
        rels = self.triples['rel'][triple_idx]
        rel_idx = np.where(rels == rel_ent)[0]
        sub_set = self.triples['sub'][triple_idx[rel_idx]]

        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    # Selecting: entSet rel > litSet
    def gen_litSet(self, entSet, rel_lit, readable=True):
        if entSet is None:
            return None

        litSet = self.gen_entSet_down(entSet, rel_lit, readable)
        p = re.compile('/[a-zA-Z0-9_]*/[a-zA-Z0-9_]*')
        if not(None in [p.match(item) for item in litSet]):
            litSet = np.array([self.obj_to_nl(item) for item in litSet])
        return litSet

    def gen_entSet_equal(self, rel_lit, value, readable=True):
        if value is None:
            return None

        if readable:
            rel_lit = self.rel2id[rel_lit]
        else:
            value = self.id2sub_obj[value]
            
        #value = value.replace(' ', '')
        triples_str = self.triples['obj_str']
        triple_idx = np.where(triples_str == value)[0]
        rels = self.triples['rel'][triple_idx]
        rel_idx = np.where(rels == rel_lit)[0]
        sub_set = self.triples['sub'][triple_idx[rel_idx]]
        
        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    def gen_entSet_atleast(self, rel_lit, value, readable=True):
        if value is None:
            return None

        if readable:
            rel_lit = self.rel2id[rel_lit]
        else:
            value = self.id2sub_obj[value]

        if type(value) != float:
            try:
                value = float(value)
            except:
                return None

        triples_num = self.triples['obj_str'][self.triples_num_idx].astype(float)
        triple_idx = np.where(triples_num <= value)[0]
        rels = self.triples['rel'][self.triples_num_idx][triple_idx]
        rel_idx = np.where(rels == rel_lit)[0]
        sub_set = self.triples['sub'][self.triples_num_idx][triple_idx[rel_idx]]

        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    def gen_entSet_less(self, rel_lit, value, readable=True):
        if value is None:
            return None

        if readable:
            rel_lit = self.rel2id[rel_lit]
        else:
            value = self.id2sub_obj[value]

        if type(value) != float:
            try:
                value = float(value)
            except:
                return None

        triples_num = self.triples['obj_str'][self.triples_num_idx].astype(float)
        triple_idx = np.where(triples_num < value)[0]
        rels = self.triples['rel'][self.triples_num_idx][triple_idx]
        rel_idx = np.where(rels == rel_lit)[0]
        sub_set = self.triples['sub'][self.triples_num_idx][triple_idx[rel_idx]]
        
        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    def gen_entSet_atmost(self, rel_lit, value, readable=True):
        if value is None:
            return None

        if readable:
            rel_lit = self.rel2id[rel_lit]
        else:
            value = self.id2sub_obj[value]

        if type(value) != float:
            try:
                value = float(value)
            except:
                return None

        triples_num = self.triples['obj_str'][self.triples_num_idx].astype(float)
        triple_idx = np.where(triples_num >= value)[0]
        rels = self.triples['rel'][self.triples_num_idx][triple_idx]
        rel_idx = np.where(rels == rel_lit)[0]
        sub_set = self.triples['sub'][self.triples_num_idx][triple_idx[rel_idx]]

        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    def gen_entSet_more(self, rel_lit, value, readable=True):
        if value is None:
            return None

        if readable:
            rel_lit = self.rel2id[rel_lit]
        else:
            value = self.id2sub_obj[value]

        if type(value) != float:
            try:
                value = float(value)
            except:
                return None

        triples_num = self.triples['obj_str'][self.triples_num_idx].astype(float)
        triple_idx = np.where(triples_num > value)[0]
        rels = self.triples['rel'][self.triples_num_idx][triple_idx]
        rel_idx = np.where(rels == rel_lit)[0]
        sub_set = self.triples['sub'][self.triples_num_idx][triple_idx[rel_idx]]
        
        result = self.np_id2sub_obj(sub_set) if readable else sub_set
        return result

    def count_entSet(self, entSet, readable=True):
        if entSet is None:
            return None

        if type(entSet) != np.ndarray:
            entSet = np.array([entSet])
        return float(len(entSet))

    def count_litSet(self, litSet, readable=True):
        if litSet is None:
            return None

        if type(litSet) != np.ndarray:
            litSet = np.array([litSet])

        return float(len(litSet))

    def maximum_litSet(self, litSet, readable=True):
        if litSet is None or type(litSet)== np.int64:
            return None

        if len(litSet) == 0:
            return None

        if not readable:
            litSet = self.np_id2sub_obj(litSet)
        try:
            litSet = litSet.astype(float)
        except:
            return None
        return max(litSet)

    def minimum_litSet(self, litSet, readable=True):
        if litSet is None or type(litSet) == np.int64:
            return None

        if len(litSet) == 0:
            return None

        if not readable:
            litSet = self.np_id2sub_obj(litSet)
        try:
            litSet = litSet.astype(float)
        except:
            return None
        return min(litSet)

    def average_litSet(self, litSet, readable=True):
        if litSet is None or type(litSet)== np.int64:
            return None

        if len(litSet) == 0:
            return None

        if not readable:
            litSet = self.np_id2sub_obj(litSet)
        try:
            litSet = litSet.astype(float)
        except:
            return None
        return np.mean(litSet)

    def intersect_entSets(self, entSet1, entSet2, readable=True):
        if entSet1 is None or entSet2 is None:
            return None

        if type(entSet1) != np.ndarray:
            entSet1 = np.array([entSet1])
        if type(entSet2) != np.ndarray:
            entSet2 = np.array([entSet2])
        entSet = np.intersect1d(entSet1, entSet2)
        return entSet

    def union_litSets(self, litSet1, litSet2, readable=True):
        if litSet1 is None or litSet2 is None:
            return None

        litSet = np.union1d(litSet1, litSet2)
        return litSet

    def union_entSets(self, entSet1, entSet2, readable=True):
        if entSet1 is None or entSet2 is None:
            return None

        if type(entSet1) != np.ndarray:
            entSet1 = np.array([entSet1])
        if type(entSet2) != np.ndarray:
            entSet2 = np.array([entSet2])
        entSet = np.union1d(entSet1, entSet2)
        return entSet

    def intersect_litSets(self, litSet1, litSet2, readable=True):
        if litSet1 is None or litSet2 is None:
            return None

        litSet = np.intersect1d(litSet1, litSet2)
        return litSet

    def concat_litSets(self, litSet1, litSet2):
        if litSet1 is None or litSet2 is None:
            return None

        return [litSet1, litSet2]

    def no_op(self, arg1, arg2, readable=False):
        return None

    def execute_trace(self, trace):
        gen_entset_down = lambda x, y: self.gen_entSet_down(x, y)
        gen_entset_up = lambda x, y: self.gen_entSet_up(x, y)
        gen_litset = lambda x, y: self.gen_litSet(x, y)
        gen_entset_equal = lambda x, y: self.gen_entSet_equal(x, y)
        gen_entset_atleast = lambda x, y: self.gen_entSet_atleast(x, y)
        gen_entset_less = lambda x, y: self.gen_entSet_less(x, y)
        gen_entset_atmost = lambda x, y: self.gen_entSet_atmost(x, y)
        gen_entset_more = lambda x, y: self.gen_entSet_more(x, y)
        count_litset = lambda x: self.count_litSet(x)
        count_entset = lambda x: self.count_entSet(x)
        maximum_litset = lambda x: self.maximum_litSet(x)
        minimum_litset = lambda x: self.minimum_litSet(x)
        average_litset = lambda x: self.average_litSet(x)
        intersect_entsets = lambda x, y: self.intersect_entSets(x, y)
        intersect_litsets = lambda x, y: self.intersect_litSets(x, y)
        union_entsets = lambda x, y: self.union_entSets(x, y)
        union_litsets = lambda x, y: self.union_litSets(x, y)
        concat_litsets = lambda x, y: self.concat_litSets(x, y)

        R = []
        for r in trace.split('<exe>')[:-1]:
            R.append(r.split('=')[-1])
        idxs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
        idxs = list(reversed(idxs[:len(R)]))
        for i, idx in enumerate(idxs):
            new_trace = R[-1] if i == 0 else new_trace.replace(f'<r{idx}>', R[-(i+1)])

        try:
            result = eval(new_trace)
        except:
            result = None
        return result


if __name__ == '__main__':
    # FOR TEST
    kg_path = f'{os.getcwd()}/data/db/mimicstar_kg/mimic_sparqlstar_kg.xml'
    kg = Graph()
    kg.parse(kg_path, format='xml', publicID='/')

    print(len(kg))
    for i, t in enumerate(kg):
        print(i, t)
        if i == 5:
            break

    q = """select * where { ?subject_id </gender> "f"^^<http://www.w3.org/2001/XMLSchema#string> }"""
    qres = kg.query(q)
    print("-" * 50)
    for res in qres:
        val = '|'
        for t in res:
            val += str(t.toPython()) + '|\t\t|'
        print(val[:-1])
    print()
    print('LOAD DONE')

    ops_path = f'{os.getcwd()}/data/db/mimicstar_kg/mimicsql_operations.json'
    interpreter = MimicInterpreter(kg_path, ops_path)

    A = interpreter.gen_litSet('/hadm_id/178264', '/admityear')
    print(A)

    st = time.time()
    q = """select ( count ( distinct ?subject_id ) as ?agg ) where { ?subject_id </hadm_id> ?hadm_id. ?hadm_id </diagnoses> ?diagnoses. ?diagnoses </diagnoses_icd9_code> ?diagnoses_icd9_code. ?diagnoses_icd9_code </diagnoses_long_title> "perforation of intestine"^^<http://www.w3.org/2001/XMLSchema#string>. ?hadm_id </lab> ?lab. ?lab </flag> "abnormal"^^<http://www.w3.org/2001/XMLSchema#string>. }"""
    qres = kg.query(q)
    print("-" * 50)
    for res in qres:
        val = '|'
        for t in res:
            val += str(t.toPython()) + '|\t\t|'
        print(val[:-1])
    et = time.time()
    print(f'time elapsed: {et - st:.2f} sec')
    print()

    st = time.time()
    A = interpreter.gen_entSet_equal(interpreter.rel2id['/flag'], 'abnormal', readable=False)
    B = interpreter.gen_entSet_up(interpreter.rel2id['/lab'], A, readable=False)
    C = interpreter.gen_entSet_up(interpreter.rel2id['/hadm_id'], B, readable=False)

    D = interpreter.gen_entSet_equal(interpreter.rel2id['/diagnoses_long_title'], 'perforation of intestine',
                                     readable=False)
    E = interpreter.gen_entSet_up(interpreter.rel2id['/diagnoses_icd9_code'], D, readable=False)
    F = interpreter.gen_entSet_up(interpreter.rel2id['/diagnoses'], E, readable=False)
    G = interpreter.gen_entSet_up(interpreter.rel2id['/hadm_id'], F, readable=False)

    H = interpreter.intersect_entSets(C, G)
    I = interpreter.count_entSet(H)
    et = time.time()
    print(f'{I} time elapsed: {et - st:.2f} sec')


