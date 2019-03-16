import numpy as np
import pandas as pd
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
import seaborn as sns
from matplotlib import pyplot as plt


molecules = pd.read_csv('data.csv', header=0)
# molecules.dropna(axis=0, inplace=True)  # drops all molecules with missing targets for predictor target analysis
aa = [Chem.MolFromSmiles(s) for s in molecules.smiles]
aa_sentences = [mol2alt_sentence(x, 1) for x in aa]
model = word2vec.Word2Vec.load('model_300dim.pkl')  # load pre-trained mol2vec
# sum mol2vec embeddings for each molecule
molecules['features'] = [DfVec(x) for x in sentences2vec(aa_sentences, model, unseen='UNK')]

rfTrees = [90, 280, 290, 260, 250, 270, 390, 230, 590, 290, 250, 430]  # optimal number of trees
lrC = [0.3, 0.9, 0.1, 0.5, 0.1, 0.4, 0.3, 0.1, 0.1, 0.1, 0.2, 0.3]  # optimal regularization coefficients
rfScores = []
lrScores = []
importances = []  # store feature importances
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for t in range(12):  # iterate over targets
    X = np.array([x.vec for x in molecules['features']])  # creates data matrix with mol2vec features
    # Code for predictor target analysis:
    # X = np.zeros((len(molecules), 311))
    # X[:, :300] = np.array([x.vec for x in molecules['features']])  # first 300 features are from mol2vec
    # Add feature for each target except t
    # j = 0
    # for i in np.delete(np.arange(12), t):
    #     X[:, 300+j] = molecules['target'+str(i+1)]
    #     j += 1
    y = np.array(molecules['target'+str(t+1)])
    notNan = ~np.isnan(y)
    X = X[notNan]
    y = y[notNan]
    y_values = []
    predictionsRf = []
    predictionsLr = []
    probasRf = []
    probasLr = []
    for train, test in kf.split(X, y):
        rf = RandomForestClassifier(n_estimators=rfTrees[t], random_state=0)
        lr = LogisticRegression(C=lrC[t])
        # Fit models
        rf.fit(X[train], y[train])
        lr.fit(X[train], y[train])
        # Predict labels for test set
        predictionsRf.append(rf.predict(X[test]))
        importances.append(rf.feature_importances_)
        predictionsLr.append(lr.predict(X[test]))
        # Class 1 probabilities
        probasRf.append(rf.predict_proba(X[test]).T[1])
        probasLr.append(lr.predict_proba(X[test]).T[1])
        y_values.append(y[test])
        del rf, lr
    # Calculate AUCs
    rfScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasRf)])
    lrScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasLr)])
# load matrix of importances of predictor targets in predicting other targets
with open('targetMatrix.pkl', 'rb') as handle:
    mat = pickle.load(handle)
# tick labels
ticks = []
for i in range(12):
    ticks.append(str(i + 1))

ax = sns.heatmap(mat, linewidth=0.5, xticklabels=ticks, yticklabels=ticks, cmap="YlGnBu")
plt.show()
