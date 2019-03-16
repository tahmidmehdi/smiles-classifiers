from __future__ import print_function
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from mol2vec.features import mol2alt_sentence
from mol2vec.helpers import depict_identifier, plot_2D_vectors, IdentifierTable, mol_to_svg


molecules = pd.read_csv('data.csv', header=0)  # load data into pandas dataframe
aa = [Chem.MolFromSmiles(s) for s in molecules.smiles]  # extract molecule objects from SMILES
aa_sentences = [mol2alt_sentence(x, 1) for x in aa]  # extract sequence of substructures (Morgan identifiers)
# Write aa_sentences to strings
# textMol is a list of strings where each element corresponds to a molecule & is a sentence of its Morgan identifiers
textMol = []
for i in range(len(aa_sentences)):
    seq = ''
    # seq will be a sentence of Morgan identifiers for substructures
    for j in range(len(aa_sentences[i])):
        seq += aa_sentences[i][j]+' '
    textMol.append(seq)
# Extract TF-IDF features from textMol
vect = TfidfVectorizer(min_df=0.01, ngram_range=(1, 5))
features = vect.fit_transform(textMol).toarray()
# Store AUCs for random forests & logistic regressions
rfScores = []
lrScores = []
rfTrees = [40, 80, 30, 120, 30, 70, 30, 60, 10, 60, 150, 20]  # optimal numbers of trees for random forests
lrC = [4, 4, 28, 49, 21, 49, 36, 34, 9, 48, 40, 49]  # optimal regularization coefficient for logistic regressions
coefs = []  # store regression coefficients
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for t in range(12):  # iterate over targets
    y = np.array(molecules['target'+str(t+1)])
    # remove molecules with NaN targets
    notNan = ~np.isnan(y)
    X = features[notNan]
    y = y[notNan]
    y_values = []
    predictionsRf = []
    predictionsLr = []
    probasRf = []
    probasLr = []
    for train, test in kf.split(X, y):  # for each fold of k-fold cross validation
        rf = RandomForestClassifier(n_estimators=rfTrees[t])
        lr = LogisticRegression(C=lrC[t])
        # Fit models
        rf.fit(X[train], y[train])
        lr.fit(X[train], y[train])
        # Store predicted labels
        predictionsRf.append(rf.predict(X[test]))
        predictionsLr.append(lr.predict(X[test]))
        coefs.append(lr.coef_)
        # Store class 1 probabilities
        probasRf.append(rf.predict_proba(X[test]).T[1])
        probasLr.append(lr.predict_proba(X[test]).T[1])
        y_values.append(y[test])  # ground-truth labels
        del rf, lr
    # Store AUCs
    rfScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasRf)])
    lrScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasLr)])
# Creates a figure of molecule 3 with the substructure 1397494279 highlighted
depict_identifier(aa[2], 1397494279, 1)
