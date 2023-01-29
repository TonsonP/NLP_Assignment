# NLP_Assignment
This repository is for submitting homework for AT82.05 Artificial Intelligence:NLU at Asian Institute of Technology

## Paper reading assignment
## 1. Evaluating Factuality in Text Simplification (ACL, 2022)
### Author Ashwin Devaraj, William Sheffield, Byron C. Wallace, Junyi Jessy Li

| Problem  | Method to evaluating models for factuality.|
| --- | --- |
| Key Related Work  | 1. Paragraph-level simplification of medical texts (Deveraj et al, 2021) -> The output from their propose new nasked language model (MLM) have hallucination*.|
|                   | 2. Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text (Laban et al, 2021) -> Text genertors predominantly introduce inaccuracies with novel entities.|
| Solution  | Construct an error annotation scheme for sentence level simplification (type of errors and their labels based on severity) and then determine the types of errors that most frequently found in various well-known simplification datasets and models.|
| Results  | In term of error from datasets |
| In term of datasets | In Newsela and Wikilarge dataset deletion errors are the most common while substitution errors are rarely occurs. Newsela dataset produce more deletion errors.|
| In term of models   | 1. Insertion error occur more often on T5 model with Newsela dataset|
|                     | 2. Most substitution erros modify single words or short phareses.|
| In term of measurement  | Current measures of semantic similarity and factuality are difficult to detect substitution errors that produce from the models|

* By providing inaccurate version of medical text (e.g. disease information) can be worse than providing no such thing at all.

## 2. A Fast and Accurate Dependency Parser using Neural Networks (EMNLP, 2014)
### Author Danqi Chen, Christopher D. Manning.

| Problem  | Dependency parsers cannot generalized well and take long time to compute. |
| --- | --- |
| Key Related Work  | 1. Deep Learning for Efficient Discriminative Parsing (Collobert, 2011) -> Proposed new NLP algorithm by using CNN with GTN for constituency parsing.|
|                   | 2. Trainsition-based Dependency Parsing Using Recursive Neural Networks (Stenetorp, 2013) -> Introduce new framework to use RvNNs for trainsition-based dependency parsing.|
|                   | 3. Natural Language Processing (almost) from Scratch (Collobert et al., 2009) -> Introduce new architecture and learning algorithm that can applied to many NLP tasks (e.g. POS, NER).
| Solution  | Combining POS tags and arc labels with words instead of discrete representation. Using Cube activation function. Use pre-computational technique to pre-compute top frequent words, all POS tags and arc labels|
| Results  | Accruacy improves around 2% in UAS and LAS scores. Improve parsing speed by 8 ~ 10 times. |

| Command | Description |
| --- | --- |
| `git status` | List all *new or modified* files |
| `git diff` | Show file differences that **haven't been** staged |
