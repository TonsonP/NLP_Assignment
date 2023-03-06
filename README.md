# NLP_Assignment
This repository is for submitting homework for AT82.05 Artificial Intelligence:NLU at Asian Institute of Technology

## Paper reading assignment
## 1. Evaluating Factuality in Text Simplification (ACL, 2022)
### Author Ashwin Devaraj, William Sheffield, Byron C. Wallace, Junyi Jessy Li

| Problem  | How to evaluating models for factuality.|
| --- | --- |
| Key Related Work  | 1. Paragraph-level simplification of medical texts (Deveraj et al, 2021) -> The output from their propose new masked language model (MLM) have hallucination*.|
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
| Solutions | By adding three-dimensional quality vectors (Lexical Diversity, Syntactic Diversity, Semantic Similarity) along with the sentences to the model (QCPG). It make the model produce target sentence that conforms to the quality constraints.
| Results  | By comparing with Baseline Model (Fine-tuned T5 model) and Gold standard (Ground-truth paraphase) with three category Semantic similarity, syntactic diversity and lexical diversity. They're measured by Bleurt, Tree edit distance and character-level edit distace. QCPG win in every category. QCPG also get better reults by human evaluation on semantic similarity. |

## 4. BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models. (ACL, 2022)
### Author Elad Ben-Zaken, Shauli Ravfogel, Yoav Goldberg

| Problem  | Fully fine-tuning models are expensive.|
| --- | --- |
| Key Related Work  | 1. Parameter-efficient transfer learning for NLP. (Houlsby et al., 2019) -> Introduce a way to fine-tuning by adding small, trainable task-specific "Adapter" modules between the layers of the pre-trained model where the original parameters are shared between tasks.|
|                   | 2. Parameter-efficient transfer learning with diff pruning. (Guo et al., 2020) -> Adding a sparse, task-specific difference-vector to original parameters, which remain fixed and are shared between tasks.|
| Solutions | Training only the bias-term and the task-specific classification.
| Results  | Reduce the parameters that need to fit by 1,000% while maintained the same accuracy but the results will be lower as the dataset get higher. |

## 5. BERT: Pre-traning of Deep Bidirectional Transformers for Language Understanding. (NAACL, 2019)
### Author Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

| Problem  | Model do not understand contextual meaning well.|
| --- | --- |
| Key Related Work  | 1. Attention is all you need. (Vaswani et al., 2019) -> Introduce a transformer models which is the foundation of BERT networks.|
|                   | 2. Cloze Procedure: A New Tool For Measuring Radability. (Taylor., 1953) -> Introduce the cloze procedure which is a teaching strategy that uses passages of text with missing word. The MLM techniques is inspired from this|
| Solutions | During pre-trained use "Masked Language Model (MLM)" to make the network learn contextual meaning<sup>1<sup> and "Next Sentence Prediction (NSP)" to make the model learn relationship between sentences<sup>2<sup>.
| Results  | Model performance improved based on GLUE testing. |

<sup>1<sup> E.g., input: "Chaky is the [mask1] at AIT, he teachs NLP [mask2]". model output: "professor", "classes".
<sup>2<sup> E.g., input: "S1: Tonson likes pizza. S2: He usually have it with pepsi" model output: "S1 then S2"

| Command | Description |
| --- | --- |
| `git status` | List all *new or modified* files |
| `git diff` | Show file differences that **haven't been** staged |
