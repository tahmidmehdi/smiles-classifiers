# SMILES Classifiers
Molecule Classifiers for InVivo AI Challenge. bagofwords.py contains code to run and cross validate random forest and logistic regressions with TF-IDF features for each molecule. mol2vec.py contains code to run and cross validate random forest and logistic regressions with latent embeddings from the mol2vec model. neuralnets.py contains code to run and cross validate convolutional recurrent neural nets (CRNNs) and RNNs for molecule classification.

## Dependencies
* Numpy
* Pandas
* sklearn
* RDKit
* Mol2vec (https://github.com/samoturk/mol2vec)
* Keras

For mol2vec.py, ensure that the model\_300dim.pkl from https://github.com/samoturk/mol2vec/tree/master/examples/models is in the same folder as the script.
