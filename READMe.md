## Description

This is a repo for training a word2vec model on domain specific data (here chemistry and material science).

First, you shoud install all the required libraries.

```
pip install -r requirements.txt
```

The goal is to learn a sophisticated word embedding for domain specific data. Also, I have used Optuna for 
hyperparameter tuning. So, It autimaticall searches in the possible space and find the best hyperparameters for 
embedding.

Generally, There are three specific parts in this repo:

1. Preprocessing data (to transform the data to the appropriate format for gensim)
2. Word embedding from scratch
3. Funetuning pre-existing model.

You can learn more about how to run (and reproduce the results) by going through `run.ipynb`. 
Please let me know if you have any questions