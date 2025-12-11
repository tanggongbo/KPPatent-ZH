# KPPatent-ZH: Chinese Patent Keyphrase Extraction and Recognition

This repository contains the dataset and code for the Chinese patent keyphrase prediction and recognition tasks, as presented in our work: **KPPatent-ZH: a new dataset and benchmarking study for keyphrase prediction in Chinese patent domain​**.

The project introduces ​**KPPatent-ZH**​, a specialized dataset for evaluating keyphrase prediction performance in the technical domain of Chinese patent documents.

## 📁 Project Structure

The repository is organized into two main directories: `data` and `code`.

| Directory   | Description                                                                                                              |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `data/` | Contains all raw and processed datasets, along with model outputs.                                                       |
| `code/` | Contains all necessary scripts for both keyphrase prediction and recognition tasks, plus configuration files. |

## 📊 Data Directory (`data/`)

This folder contains all data files essential for reproducing our experiments:

* **original_data:** The ready_to_use data.
* **Keyphrase Prediction Data:** Processed data formatted for the keyphrase prediction task.
* **Keyphrase Recognition Data:** Data specifically curated for the discriminative tasks, including the **Single-Choice Question (SCQ), True/False (TF), and Multiple-Choice Question (MCQ)** sets with hard negative keyphrases. 

## 💻 Code Directory (`code/`)

This folder holds the scripts required to run and replicate the experiments:

* **prediction:** Scripts for running both the extractive and generative **keyphrase prediction** task.
* **recognition:** Code for the **keyphrase recognition** (SCQ/TF/MCQ) tasks. 

## 🙏 Acknowledgements

We sincerely thank the developers of the following outstanding open-source projects, which were instrumental in our research and evaluation:

* **KeyLLM:** A minimal method for keyword extraction with Large Language Models (LLM).
  * **URL:** [https://maartengr.github.io/KeyBERT/guides/keyllm.html#1-create-keywords-with-keyllm](https://maartengr.github.io/KeyBERT/guides/keyllm.html#1-create-keywords-with-keyllm "null")
* **pke-zh:** Python Keyphrase Extraction for zh(chinese).
  * **URL:** [https://github.com/shibing624/pke\_zh](https://github.com/shibing624/pke_zh "null")
* **KPEval:** KPEval is a toolkit for evaluating keyphrase related systems.
  * **URL:** [https://github.com/uclanlp/KPEval](https://github.com/uclanlp/KPEval "null")

