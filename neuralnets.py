import numpy as np
import pandas as pd
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def f1(y_true, y_pred):
    """
    Calculates F1 score from predicted labels and true labels.
    Written by Paddy & Kev1n91 from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

    :param list-like y_true: actual labels
    :param list-like y_pred: predicted labels
    :return: f1 score
    """

    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


np.random.seed(0)  # set seed for reproducibility

molecules = pd.read_csv('data.csv', header=0)
aa = [Chem.MolFromSmiles(s) for s in molecules.smiles]
aa_sentences = [mol2alt_sentence(x, 1) for x in aa]
vocab = np.unique([x for l in aa_sentences for x in l])  # array of unique substructures (Morgan identifiers)
numWords = len(vocab)  # number of unique substructures
# Create a mapping of Morgan identifiers to integers between 1 and numWords
wordMap = dict()
for i in range(len(vocab)):
    wordMap[vocab[i]] = i+1
# aa_map is like aa_sentences but Morgan identifiers are replaced by their value in wordMap
aa_map = []
for m in aa_sentences:
    aa_map.append([wordMap[s] for s in m])
aa_map = np.array(aa_map)

maxSeqLen = max([len(aa_map[i]) for i in range(len(aa_map))])  # length of longest molecule
embedding_length = 32

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
crnnScores = []
rnnScores = []
# Create CRNN
crnn = Sequential()
crnn.add(Embedding(numWords+1, embedding_length, input_length=maxSeqLen))
crnn.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
crnn.add(MaxPooling1D(pool_size=3))
crnn.add(LSTM(100))
crnn.add(Dense(1, activation='sigmoid'))
crnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
print(crnn.summary())
# Create RNN
rnn = Sequential()
rnn.add(Embedding(numWords+1, embedding_length, input_length=maxSeqLen))
rnn.add(LSTM(100))
rnn.add(Dense(1, activation='sigmoid'))
rnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
print(rnn.summary())
for t in range(12):  # iterate over targets
    y = molecules['target'+str(t+1)]
    notNan = ~np.isnan(y)
    y = np.array([int(i) for i in y[notNan]])
    X = aa_map[notNan]
    y_values = []
    predictionsCrnn = []
    predictionsRnn = []
    probasCrnn = []
    probasRnn = []
    for train, test in kf.split(X, y):
        # Make sequences have same length with 0 padding on both sides
        X_train = sequence.pad_sequences(X[train], maxlen=maxSeqLen)
        X_test = sequence.pad_sequences(X[test], maxlen=maxSeqLen)
        # Fit CRNN
        crnn.fit(X_train, y[train], validation_data=(X_test, y[test]), epochs=3, batch_size=64)
        # predict class 1 probabilities for test set
        probs = crnn.predict(X_test).reshape(len(X_test))
        # predict label from probs
        preds = np.round(probs)
        preds = np.array([int(i) for i in preds])
        predictionsCrnn.append(preds)
        probasCrnn.append(probs)
        # Fit RNN
        rnn.fit(X_train, y[train], validation_data=(X_test, y[test]), epochs=3, batch_size=64)
        probs = rnn.predict(X_test).reshape(len(X_test))
        preds = np.round(probs)
        preds = np.array([int(i) for i in preds])
        predictionsRnn.append(preds)
        probasRnn.append(probs)
        y_values.append(y[test])
    # Store AUCs
    crnnScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasCrnn)])
    rnnScores.extend([roc_auc_score(y, proba) for y, proba in zip(y_values, probasRnn)])
