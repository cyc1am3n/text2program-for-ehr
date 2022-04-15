# Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records (CHIL 2022)
by **Daeyoung Kim (KAIST), Seongsu Bae (KAIST), Seungho Kim (KAIST), Edward Choi (KAIST)**

This repository provides the official implementation of the [Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records](https://arxiv.org/abs/2203.06918).

## Requirements
- PyTorch == 1.7.1
- Python == 3.8.5
- transformers == 4.5.1
- numpy == 1.19.5
- pytorch-lightning == 1.3.2
- rdflib == 5.0.0

## Data
### Prepare Knowledge Graph
You should build knowledge graph for MIMICSPARQL* following instruction in [official MIMICSPARQL* github](https://github.com/junwoopark92/mimic-sparql).  
The KG(`mimic_sparqlstar_kg.xml`) file should be in `./data/db/mimicstar_kg` directory.

### Pre-process
Generate dictionary files for the recovery technique.
```shell script
$ cd data
$ python preprocess.py
```

## Train
```shell script
$ python main.py
```

## Test
```shell script
$ python main.py --test
```

## Citation
```
@article{kim2022uncertainty,
  title={Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records},
  author={Kim, Daeyoung and Bae, Seongsu and Kim, Seungho and Choi, Edward},
  journal={arXiv preprint arXiv:2203.06918},
  year={2022}
}
```