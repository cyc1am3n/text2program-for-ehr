import re

REL_TERM_TO_NL = {'/itemid': 'item id', '/charttime': 'lab test chart time', '/flag': 'lab test abnormal status',
                  '/value_unit': 'lab test value', '/route': 'drug route', '/drug_type': 'drug type',
                  '/icustay_id': 'icu stay id', '/drug_dose': 'drug dose', '/formulary_drug_cd': 'drug code', '/drug': 'drug name',
                  '/lab': 'lab id',  '/discharge_location': 'discharge location',
                  '/prescriptions': 'prescriptions id', '/admityear': 'admission year', '/admittime': 'admission time',
                  '/insurance': 'insurance', '/dischtime': 'discharge time', '/diagnosis': 'primary disease',
                  '/days_stay': 'days of hospital stay', '/admission_type': 'admission type', '/marital_status': 'marital status',
                  '/religion': 'religion', '/admission_location': 'admission location', '/age': 'age', '/language': 'language',
                  '/ethnicity': 'ethnicity', '/fluid': 'lab test fluid', '/label': 'lab test name',
                  '/category': 'lab test category', '/diagnoses_short_title': 'diagnoses short title', '/diagnoses_long_title': 'diagnoses long title',
                  '/procedures_short_title': 'procedure short title', '/procedures_long_title': 'procedure long title',
                  '/diagnoses_icd9_code': 'diagnoses icd9 code', '/dob_year': 'year of birth', '/dod_year': 'year of death',
                  '/gender': 'gender', '/name': 'subject name', '/dob': 'date of birth', '/expire_flag': 'death status',
                  '/dod': 'date of death', '/procedures_icd9_code': 'procedure icd9 code', '/diagnoses': 'diagnoses id',
                  '/hadm_id': 'hospital admission id', '/procedures': 'procedures id'}
ENTITY = ['subject id', 'hadm id', 'lab', 'diagnoses', 'procedures', 'prescriptions', 'itemid', 'diagnoses icd9 code', 'procedures icd9 code']

def extract_semantic_from_template(template):
    for rel_term, nl in REL_TERM_TO_NL.items():
        template = template.replace(f" {nl} ", f" {rel_term} ")
    if 'how many' in template or 'count the number' in template or 'provide the number' in template or \
        'give me the number' in template or 'what is the number' in template:
        if 'how many' in template:
            question_template = 'how many'
        elif 'count the number' in template:
            question_template = 'count the number of'
        elif 'provide the number' in template:
            question_template = 'provide the number of'
        elif 'give me the number' in template:
            question_template = 'give me the number of'
        elif 'what is the number' in template:
            question_template = 'what is the number of'
        match = list(re.findall(fr'{question_template} (.*) whose (.*) is (.*) and (.*) is (.*)[\?|\.]', template))
        if len(match) != 0:
            q_type = 6
            match = list(match[0])
            if ' is ' in match[1]:
                except_is = match[1].split(' is ')
                assert len(except_is) < 3
                match[1] = except_is[0]
                match[2] = except_is[1] + ' is ' + match[2] 
            if ' is ' in match[3]:
                except_is = match[3].split(' is ')
                assert len(except_is) < 3
                match[3] = except_is[0]
                match[4] = except_is[1] + ' is ' + match[4]
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[2].startswith(condition):
                    match.insert(2, 'is '+condition[:-1])
                    match[3] = match[3].replace(condition, '')
                    break
                if i == 3:
                    match.insert(2, 'is')
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[-1].startswith(condition):
                    match.insert(-1, 'is ' + condition[:-1])
                    match[-1] = match[-1].replace(condition, '')
                    break
                if i == 3:
                    match.insert(-1, 'is')
        else:
            q_type = 5
            match = list(re.findall(fr'{question_template} (.*) whose (.*) is (.*)[\?|\.]', template)[0])
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[2].startswith(condition):
                    match.insert(2, 'is '+ condition[:-1])
                    match[-1] = match[-1].replace(condition, '')
                    break
                if i == 3:
                    match.insert(2, 'is')
    elif 'what is maximum' in template or 'what is minimum' in template or 'what is average' in template:
        match = list(re.findall(r'what is (.*) of patients whose (.*) is (.*) and (.*) is (.*)\?', template))
        if len(match) != 0:
            q_type = 8
            match = list(match[0])
            numeric_literal = match[0].split()
            match[0] = numeric_literal[0]
            match.insert(1, ' '.join(numeric_literal[1:]))
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[3].startswith(condition):
                    match.insert(3, 'is '+condition[:-1])
                    match[4] = match[4].replace(condition, '')
                    break
                if i == 3:
                    match.insert(3, 'is')
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[-1].startswith(condition):
                    match.insert(-1, 'is '+condition[:-1])
                    match[-1] = match[-1].replace(condition, '')
                    break
                if i == 3:
                    match.insert(-1, 'is')
        else:
            q_type = 7
            match = list(re.findall(r'what is (.*) of patients whose (.*) is (.*)\?', template)[0])
            numeric_literal = match[0].split()
            match[0] = numeric_literal[0]
            match.insert(1, ' '.join(numeric_literal[1:]))
            for i, condition in enumerate(['less than or equal to ', 'greater than or equal to ', 'less than ', 'greater than ']):
                if match[3].startswith(condition):
                    match.insert(3, 'is '+condition[:-1])
                    match[-1] = match[-1].replace(condition, '')
                    break
                if i == 3:
                    match.insert(3, 'is')
    else:
        match = list(re.findall(r'what is (.*) of (.*)\?', template)[0])
        if 'of' in match[0]:
            of_split = ' of '.join(match).split(' of ')
            match = [of_split[0], ' of '.join(of_split[1:])]
        #if ('subject id' in match[1] or 'diagnoses icd9 code' in match[1] or 'procedure icd9 code' in match[1] or 'hadm id' in match[1] or 'lab id' in match[1] or 'prescription'):
        if list(filter(None, [re.fullmatch(fr'{entity} [a-zA-Z0-9]*', match[1]) for entity in ENTITY])):
            if 'and' in match[0]:
                q_type = 2
                litsets = match[0].split(' and ')
                entity = match[1]
                match[0] = litsets[0]
                match.insert(1, litsets[1])
            else:
                q_type = 1
        else:
            if 'and' in match[0]:
                q_type = 4
                litsets = match[0].split(' and ')
                for rel_term in REL_TERM_TO_NL.keys():
                    if rel_term in match[1]:
                        literal = rel_term
                        break
                value = match[1].replace(literal+' ', '')
                match[0] = litsets[0]
                match[1] = litsets[1]
                match.append(literal)
                for i, condition in enumerate(['is less than or equal to ', 'is greater than or equal to ', 'is less than ', 'is greater than ']):
                    if value.startswith(condition):
                        match.append(condition[:-1])
                        value = value.replace(condition, '')
                        break
                    if i == 3:
                        match.append('is')
                match.append(value)
            else:
                q_type = 3
                for rel_term in REL_TERM_TO_NL.keys():
                    if rel_term in match[1]:
                        literal = rel_term
                        break
                match[0] = match[0]
                value = match[1].replace(literal+' ', '')
                match[1] = literal
                for i, condition in enumerate(['is less than or equal to ', 'is greater than or equal to ', 'is less than ', 'is greater than ']):
                    if value.startswith(condition):
                        match.append(condition[:-1])
                        value = value.replace(condition, '')
                        break
                    if i == 3:
                        match.append('is')
                match.append(value)
    for i, item in enumerate(match):
        if item in REL_TERM_TO_NL.keys():
            match[i] = REL_TERM_TO_NL[item]
    semantic = [f"[type{q_type}]"] + match
    return '<t>'.join(semantic)