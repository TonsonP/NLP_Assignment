# NLP_Assignment
This repository is for submitting homework for AT82.05 Artificial Intelligence:NLU at Asian Institute of Technology

## Paper reading assignment
## 1. Evaluating Factuality in Text Simplification (ACL, 2022)
### Author Ashwin Devaraj, William Sheffield, Byron C. Wallace, Junyi Jessy Li

| Problem  | How to evaluating models for factuality.|
| --- | --- |
| Key Related Work  | 1. Paragraph-level simplification of medical texts (Deveraj et al, 2021) -> The output from their propose new nasked language model (MLM) have hallucination*.|
|                   | 2. Keep It Simple: Unsupervised Simplification of Multi-Paragraph Text (Laban et al, 2021) -> Text simplification model introduces false information.|
| Solution  | Construct an error annotation scheme for sentence level simplification (type of errors and their labels based on severity) and then determine the types of errors that most frequently found in various well-known simplification datasets and models.|
| Results  ||
| In term of datasets | In Newsela and Wikilarge dataset deletion errors are the most common while substitution errors are rarely occurs. Newsela dataset produce more deletion errors.|
| In term of models   | Insertion error occur more often on T5 model with Newsela dataset. Substitution error mostly caused by model rather than dataset|
| In term of measurement  | Current measures of semantic similarity and factuality are difficult to detect substitution errors that produces from the model|

* Personal Noted: By providing inaccurate version of medical text (e.g. disease information) can be worse than providing no such thing at all.

## 2. A Fast and Accurate Dependency Parser using Neural Networks (EMNLP, 2014)
### Author Danqi Chen, Christopher D. Manning.

| Problem  | Dependency parsers cannot generalized well and take long time to compute. |
| --- | --- |
| Key Related Work  | 1. Deep Learning for Efficient Discriminative Parsing (Collobert, 2011) -> Proposed new NLP algorithm by using CNN with GTN for constituency parsing.|
|                   | 2. Trainsition-based Dependency Parsing Using Recursive Neural Networks (Stenetorp, 2013) -> Introduce new framework to use RvNNs for trainsition-based dependency parsing.|
|                   | 3. Natural Language Processing (almost) from Scratch (Collobert et al., 2009) -> Introduce new architecture and learning algorithm that can applied to many NLP tasks (e.g. POS, NER).
| Solution  | Combining POS tags and arc labels with words instead of discrete representation. Using Cube activation function. Use pre-computational technique to pre-compute top frequent words, all POS tags and arc labels|
| Results  | Accruacy improves around 2% in UAS and LAS scores. Improve parsing speed by 8 ~ 10 times. |

## 3. Quality Controlled Paraphrase Generation. (ACL, 2022)
### Author Elron Bandel, Rannit Aharonov, Michal Shmueli-Scheuer, Ilya Shnayderman, Noam Slonim, Liat Ein-Dor

| Problem  | How can we control output quality of text-to-text model.|
| --- | --- |
| Key Related Work  | 1. The components of paraphrase evaluations (McCarthy et al., 2009) -> Discovered that semantic and syntactic evaluations are the primary components of paraphase quality.|
|                   | 2. Adversarial Example Generation with Syntactically Controlled Paraphase Networks (Iyyer et al., 2018) -> Introduce a way to predict syntactic tree which is the groundwork of normalized tree edit distance that are used in this paper.|
|                   | 3. BLEURT (Sellam et al., 2020) -> Introduce evaluation metric for Natural Language Generation which have high correlation with human judge.|
| Results  | By comparing with Baseline Model (Fine-tuned T5 model) and Gold standard (Ground-truth paraphase) with three category Semantic similarity, syntactic diversity and lexical diversity. They're measured by Bleurt, Tree edit distance and character-level edit distace. QCPG win in every category. QCPG also get better reults by human evaluation on semantic similarity. |

| Command | Description |
| --- | --- |
| `git status` | List all *new or modified* files |
| `git diff` | Show file differences that **haven't been** staged |
