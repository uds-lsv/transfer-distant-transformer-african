Data for and/or by the EMNLP'20 paper

Transfer Learning and Distant Supervision for Multilingual Transformer Models: 
A Study on African Languages

The data for each experiment has the same structure:

- train_clean.tsv - main training data in the target language
- train_noisy.tsv - additionally, distantly supervised training data (if used)
- train_clean_noisy_labels.tsv - version of train_clean with distantly supervised labels (needed by the cm noise handling model)
- dev.tsv - development set
- test.tsv - test set

All files are tab separated (tsv). The first column is the token (for NER) or sentence (for topic classification). For NER,
a sentence border is indicated by an empty line.

The original Yoruba NER data was obtained from https://github.com/ajesujoba/YorubaTwi-Embedding/tree/master/Yoruba/Yor%C3%B9b%C3%A1-NER

The isiXhosa NER data can be obtained from https://repo.sadilar.org/handle/20.500.12185/312
The AG News corpus can be obtained here: https://data.wluper.com/
The CoNLL03 corpus can be obtained here: https://www.clips.uantwerpen.be/conll2003/ner/

For using the datasets from other authors (isiXhosa, AG News, CoNLL03) with our code, you have to 
rename the files to fit the schema above (filename train_clean.tsv, etc.). You can use prepare_ag_data.py
to convert the AG News data into the expected tsv format.

For more information about the data creation as well as the data sources, please refer to the Appendix
in the paper.

