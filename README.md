# NLP_Assignment
This repository is for submitting homework for AT82.05 Artificial Intelligence:NLU at Asian Institute of Technology

## Paper reading assignment
## 1. Evaluating Factuality in Text Simplification (ACL, 2022)
### Author Ashwin Devaraj, William Sheffield, Byron C. Wallace, Junyi Jessy Li

| Problem  | Second Header |
| --- | --- |
| Key Related Work  | swwwwwww  |
| Solution  | Content Cell  |
| Results  | Content Cell  |


## 2. A Fast and Accurate Dependency Parser using Neural Networks (EMNLP, 2014)
### Author Danqi Chen, Christopher D. Manning.

| Problem  | Dependency parsers cannot generalized well and take long time to compute. |
| --- | --- |
| Key Related Work  | 1. Deep Learning for Efficient Discriminative Parsing (Collobert, 2011) -> Proposed new NLP algorithm by using CNN with GTN for constituency parsing.|
|                   | 2. Trainsition-based Dependency Parsing Using Recursive Neural Networks (Stenetorp, 2013) -> Introduce new framework to use RvNNs for trainsition-based dependency parsing.|
|                   | 3. Natural Language Processing (almost) from Scratch (Collobert et al., 2009) -> Introduce new architecture and learning algorithm that can applied to many NLP tasks (e.g. POS, NER).
| Solution  | Combining POS tags and arc labels with words instead of discrete representation. Using Cube activation function. Use pre-computational technique to pre-compute top frequent words, all POS tags and arc labels|
| Results  | Accruacy improves around 2% in UAS and LAS scores. Improve parser speed by 8 ~ 10 times. |

| Command | Description |
| --- | --- |
| `git status` | List all *new or modified* files |
| `git diff` | Show file differences that **haven't been** staged |
