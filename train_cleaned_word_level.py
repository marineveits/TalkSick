#!/usr/bin/python


########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import operator
import sys
import pickle

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import sotoxic.models.keras.model_zoo as model_zoo
from sotoxic.train import trainer
from sotoxic.data_helper.data_loader import DataLoader



EMBEDDING_FILE='crawl-300d-2M.vec'
TRAIN_DATA_FILE='train.csv'
TEST_DATA_FILE='test.csv'

MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300


train_df = pd.read_csv(TRAIN_DATA_FILE)
test_df = pd.read_csv(TEST_DATA_FILE)
data_loader = DataLoader()
embeddings_index = data_loader.load_embedding(EMBEDDING_FILE)


########################################
## process texts in datasets
########################################
print('Processing text dataset')
list_sentences_train = train_df["comment_text"].fillna("no comment").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = train_df[list_classes].values
list_sentences_test = test_df["comment_text"].fillna("no comment").values
y_test = test_df[list_classes].values
comments = []
for text in list_sentences_train:
    comments.append(text)
    
test_comments=[]
for text in list_sentences_test:
    test_comments.append(text)

#tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='"#%&()+,-./:;<=>@[\\]^_`{|}~\t\n')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(comments + test_comments)

sequences = tokenizer.texts_to_sequences(comments)
test_sequences = tokenizer.texts_to_sequences(test_comments)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y_train.shape)

test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of test_data tensor:', test_data.shape)
print('Shape of label tensor:', y_test.shape)


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))
#embedding_matrix = np.random.normal(loc=matrix_mean, scale=matrix_std, size=(nb_words, EMBEDDING_DIM))
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
null_count = 0
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        null_count += 1
print('Null word embeddings: %d' % null_count)

with open('embedding_matrix.pkl', 'wb') as picklefile:
        pickle.dump(embedding_matrix, picklefile)
        
### Training
def get_model():
    return model_zoo.get_kmax_text_cnn(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, out_size=6)

keras_model_trainer = trainer.KerasModelTrainer(model_stamp='kmax_text_cnn', epoch_num=20, learning_rate=1e-3)


models, val_loss, total_auc, fold_predictions = keras_model_trainer.train_folds(data, y_train, fold_count=2, batch_size=256, get_model_func=get_model)

### Save best model
model=get_model()
model.load_weights('kmax_text_cnn1.h5')
model.save('best_model_final1.h5')


print("Overall val-loss:", val_loss, "AUC", total_auc)


### Predections

train_fold_preditcions = np.concatenate(fold_predictions, axis=0)


training_auc = roc_auc_score(y_train, train_fold_preditcions)
print("Training AUC", training_auc)


CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submit_path_prefix = "fasttext-SC2-nds-randomNoisy-capNet-" + str(MAX_NB_WORDS) + "-RST-lp-ct-" + str(MAX_SEQUENCE_LENGTH) 

print("Predicting testing results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    test_predicts = model.predict(test_data, batch_size=256, verbose=1)
    test_predicts_list.append(test_predicts)
    np.save("predict_path/", test_predicts)

test_predicts = np.zeros(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts += fold_predict
test_predicts /= len(test_predicts_list)

with open('test_predicts.pkl', 'wb') as picklefile:
        pickle.dump(test_predicts, picklefile)

test_predicts=test_predicts >= 0.5

correct_test = np.where(test_predicts==y_test)[0]
print ("Found %d correct labels" % len(correct_test))
incorrect_test = np.where(test_predicts!=y_test)[0]
print ("Found %d incorrect labels" % len(incorrect_test))

lst_test_predicts=test_predicts.ravel().tolist()
lst_y_test=y_test.ravel().tolist()

tp=0
tn=0
fp=0
fn=0
for i in range(len(lst_test_predicts)):
    if lst_test_predicts[i]==True and lst_y_test[i]==1:
        tp+=1
    elif lst_test_predicts[i]==False and lst_y_test[i]==0:
        tn+=1
    elif lst_test_predicts[i]==True and lst_y_test[i]==0:
        fp+=1
    else:
        fn+=1
print('tp:',tp,'tn:',tn,'fp:',fp,'fn:',fn)

precision=tp/(tp+fp)
print('Precision:', precision)
recall=tp/(tp+fn)
print('Recall:', recall)
f1=2*((precision*recall)/(precision+recall))
print('F1:', f1)
f5=(1+5**2)*((precision*recall)/((5**2*precision)+recall))
print('F5:', f5)

test_ids = test_df["id"].values
test_ids = test_ids.reshape((len(test_ids), 1))

test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-L{:4f}-A{:4f}.csv".format(val_loss, total_auc)
test_predicts.to_csv(submit_path, index=False)




print("Predicting training results...")

train_ids = train_df["id"].values
train_ids = train_ids.reshape((len(train_ids), 1))

train_predicts = pd.DataFrame(data=train_fold_preditcions, columns=CLASSES)
train_predicts["id"] = train_ids
train_predicts = train_predicts[["id"] + CLASSES]
submit_path = submit_path_prefix + "-Train-L{:4f}-A{:4f}.csv".format(val_loss, training_auc)
train_predicts.to_csv(submit_path, index=False)

