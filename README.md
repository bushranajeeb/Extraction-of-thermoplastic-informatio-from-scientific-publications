# Extraction-of-thermoplastic-informatio-from-scientific-publications

The script to train the model on annotated dataset is model_training.py

Data scraping steps used for this master thesis are in the jupyter notebook data_scraping.ipynb

## Data Extraction Script

The script for extracting data using the model combined with heuristic rules can be run by using the following command:

```bash
python data_extraction.py --model_name <model_name> --pdf_file <path_to_pdf_file>
```
Example Usage:
```bash
python data_extraction.py --model_name bert_ner_praqs --pdf_file data/test.pdf
```

The components of the script are in following files:
1- token_processor.py: TokenProcessor class handles token processing, including Sub-token combination, Detection of blended material terms, Processing of delimiters (/, +, -) in material names.
2- material_processor.py: MaterialProcessor class extracts materials, their reinforcing materials, their property value pairs, and comparisons
3- core_components.py: It contains core classes like SentenceProcessing for sentence parsing and AutomotivePartFinder for automotive part extraction
4- material_property_finder.py: MaterialPropertyFinder Class extracts material properties and their values

preprocessing.py can be used to preprocess the whole corpus. some componnents of the class are also called in data_extraction script to load and clean the pdf before extraction.
