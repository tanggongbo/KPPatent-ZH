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

* **original_data:** The ready_to_use data).
* **Keyphrase Prediction Data:** Processed data formatted for the keyphrase prediction task (e.g., training, validation, and testing splits).
* **Keyphrase Recognition Data:** Data specifically curated for the discriminative tasks, including the **Single-Choice Question (SCQ), True/False (TF), and Multiple-Choice Question (MCQ)** sets with hard negative keyphrases. 

## 💻 Code Directory (`code/`)

This folder holds the scripts required to run and replicate the experiments:

* **prediction:** Scripts for running both the extractive and generative **keyphrase prediction** task.
* **recognition:** Code for the **keyphrase recognition** (SCQ/TF/MCQ) tasks. 


## License & Data Compliance

This repository employs a dual-licensing framework to cover both software code and processed datasets:

1. **Source Code**:
   All source code, scripts, and model pipelines in this repository are licensed under the [MIT License](LICENSE).

2. **Dataset & Artifacts**:
   All datasets and model outputs (located in the `data/` directory and the datasets permanently archived on Zenodo at [DOI: 10.5281/zenodo.22702877](https://zenodo.org/records/22702877)) are licensed under the **[Creative Commons Attribution 4.0 International (CC BY 4.0)](data/DATA_LICENSE)** license.

3. **Data Source, Transformation, and Legal Compliance**:
   - **Original Data Source**: The raw materials originate from the [Patent Search and Analysis System of the China National Intellectual Property Administration (CNIPA)](https://pss-system.cponline.cnipa.gov.cn/notice). The intellectual property rights of the raw data belong to CNIPA.
   - **Data Transformation & Redistribution**: In strict compliance with CNIPA's copyright terms (which prohibit commercial reproduction in original form and layout), this dataset does **not** redistribute verbatim documents or original-format mirrors:
     - **Patent Texts**: Raw patent texts have been cleaned, filtered, and restructured into processed analytical schemas.
     - **Examiner Search-Report Information**: All original narrative texts, official document layouts, and correspondence have been excluded. Only abstract structured mappings—specifically, the patent ID and its associated keyphrase list—are retained and merged into standardized entries.
   - **Usage & Disclaimer**: This dataset is provided for academic research and educational purposes. Users must give appropriate credit to the original authors and adhere to the source provider's terms of use.


## 🙏 Acknowledgements

We sincerely thank the developers of the following outstanding open-source projects, which were instrumental in our research and evaluation:

* **KeyLLM:** A minimal method for keyword extraction with Large Language Models (LLM).
  * **URL:** [https://maartengr.github.io/KeyBERT/guides/keyllm.html#1-create-keywords-with-keyllm](https://maartengr.github.io/KeyBERT/guides/keyllm.html#1-create-keywords-with-keyllm "null")
* **pke-zh:** Python Keyphrase Extraction for zh(chinese).
  * **URL:** [https://github.com/shibing624/pke\_zh](https://github.com/shibing624/pke_zh "null")
* **KPEval:** KPEval is a toolkit for evaluating keyphrase related systems.
  * **URL:** [https://github.com/uclanlp/KPEval](https://github.com/uclanlp/KPEval "null")


