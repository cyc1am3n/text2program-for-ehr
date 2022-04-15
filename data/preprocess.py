import json
from collections import defaultdict
from rdflib import Graph

from transformers import AutoTokenizer

if __name__ == "__main__":
    kg_path = './db/mimicstar_kg/mimic_sparqlstar_kg.xml'
    kg = Graph()
    kg.parse(kg_path, format='xml', publicID='/')
    print("KG Loaded.")

    sub, rel, obj = [], [], []
    rel_obj, sub_rel, sub = defaultdict(set), defaultdict(set), defaultdict(set)
    for t in kg:
        sub_rel[f"/{t[0].toPython().split('/')[1]}"].add(t[1].toPython())
        sub[f"/{t[0].toPython().split('/')[1]}"].add(t[0].toPython())
        rel_obj[t[1].toPython()].add(str(t[2].toPython()))
        
    sub_dict = {k: list(v) for k, v in sub.items()}
    sub_rel_dict = {k: list(v) for k, v in sub_rel.items()}
    rel_obj_dict = {k: list(v) for k, v in rel_obj.items()}

    exclude_keys = ['/value_unit', '/drug_dose', '/charttime', '/dischtime', '/dob_year', '/admityear', '/expire_flag', '/age',
               '/dod', '/admittime', '/dob', '/days_stay', '/dod_year',
               '/icustay_id', '/itemid', '/lab', '/prescriptions', '/diagnoses', '/diagnoses_icd9_code', '/procedures' ,'/procedures_icd9_code', '/hadm_id']

    filtered_rel_obj_dict = {}
    for key in list(rel_obj_dict.keys()):
        if not key in exclude_keys:
            filtered_rel_obj_dict[key] = rel_obj_dict[key]

    with open(f'./rel_obj_look_up.json', 'w') as f:
        f.write(json.dumps(filtered_rel_obj_dict))
    print("Dictionary of relation-object is saved.")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    data = {}
    for key in rel_obj_dict.keys():
        for sample in rel_obj_dict[key]:
            data[' '.join(tokenizer.tokenize(sample))] = sample
    
    with open(f'./cond_look_up.json', 'w') as f:
        f.write(json.dumps(data))
    print("Dictionary for handling tokenization issue of bert tokenizer is saved.")