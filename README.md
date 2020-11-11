# Transfer Learning and Distant Supervision for Multilingual Transformer Models: A Study on African Languages

## Introduction

This is the repository for the publication

> Michael A. Hedderich, David Adelani, Dawei Zhu, Jesujoba Alabi, Udia Markus & Dietrich Klakow 
>
> **Transfer Learning and Distant Supervision for Multilingual Transformer Models: A Study on African Languages**
>
> EMNLP 2020
>
> https://www.aclweb.org/anthology/2020.emnlp-main.204/

This repository contains

- the newly created datasets
- the code for the Transformer experiments
- a full table with all the results

## Datasets
For this work, we created three new datasets
- Hausa NER
- Hausa topic classification
- Yoruba topic classification

They can be found in the */data* directory. For additional information on the data creation process, please refer to the paper (especially the Appendix).

## Code
The code for the Transformer models (both for NER and topic classification) can be found in the */code/transformer* directory.
### Dependencies
Dependency managment is provided via [Singularity](https://sylabs.io/docs/) (similar to Docker). Please use the */code/transformer/singularity_dependencies.def* file. If you do not want to use Singularity, you can also manually install the packages with the versions specified in the Singularity file.
### Running the Code
Both the *ner* and *topic* directories contain a *run.py* file respectively. This file specifies all the experimental configurations reported in this work. Call the experiment method you want to run at the bottom of the file and just execute
``
python run.py
``
within the Singularity container. You will have to download the data beforehand and change the paths in the configurations to match the settings on your machine (those marked in the code with *#CHANGE_ME*).

## Results
We provide a table with the experimental scores including F1-scores for each run as well as the mean and standard error. This table can be found in */results/results.csv*.

## Contact & Citation
If you have any questions or encounter any issues, feel free to contact the first author (Michael; email address in the paper). 

If you use this work, please cite us as

```
@inproceedings{hedderich-etal-2020-transfer,
    title = "Transfer Learning and Distant Supervision for Multilingual Transformer Models: A Study on {A}frican Languages",
    author = "Hedderich, Michael A.  and
      Adelani, David  and
      Zhu, Dawei  and
      Alabi, Jesujoba  and
      Markus, Udia  and
      Klakow, Dietrich",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.204",
    pages = "2580--2591",
}
}
```
