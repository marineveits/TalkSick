#!/usr/bin/python

from slackclient import SlackClient

import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import tensorflow as tf
import operator
import sys
import pickle
import nltk
import time
import random
import h5py

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, merge, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers import merge
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K


import sotoxic.models.keras.model_zoo as model_zoo
from sotoxic.train import trainer
from sotoxic.data_helper.data_loader import DataLoader



# ------------------------------------------------------------------------------
# LOAD MY MODEL
# ------------------------------------------------------------------------------


with open('word_index.pkl', 'rb') as file:
    word_index=pickle.load(file)

with open('embedding_matrix.pkl', 'rb') as file:
    embedding_matrix=pickle.load(file)    

MAX_SEQUENCE_LENGTH = 400
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_index))

def get_kmax_text_cnn(nb_words, embedding_dim, embedding_matrix, max_sequence_length, out_size):
    embedding_layer = Embedding(nb_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length,
                                trainable=False)

    filter_nums = 180
    drop = 0.6

    comment_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(comment_input)
    embedded_sequences = SpatialDropout1D(0.2)(embedded_sequences)

    conv_0 = Conv1D(filter_nums, 1, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_1 = Conv1D(filter_nums, 2, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_2 = Conv1D(filter_nums, 3, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)
    conv_3 = Conv1D(filter_nums, 4, kernel_initializer="normal", padding="valid", activation="relu")(embedded_sequences)

    maxpool_0 = model_zoo.KMaxPooling(k=3)(conv_0)
    maxpool_1 = model_zoo.KMaxPooling(k=3)(conv_1)
    maxpool_2 = model_zoo.KMaxPooling(k=3)(conv_2)
    maxpool_3 = model_zoo.KMaxPooling(k=3)(conv_3)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3], axis=1)
    output = Dropout(drop)(merged_tensor)
    output = Dense(units=144, activation='relu')(output)
    output = Dense(units=out_size, activation='sigmoid')(output)

    model = Model(inputs=comment_input, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, decay=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model



# ------------------------------------------------------------------------------
# Slackbot code
# ------------------------------------------------------------------------------

# Our Slack officer is called TalkSick and here are his credentials:
bot_token = os.environ.get('TalkSick_TOKEN')
bot_name='TalkSick'
slack_client = SlackClient(bot_token)
# Initialize the Slack client and find out its ID 
# (so we can filter its messages):
users = slack_client.api_call("users.list").get('members')
for user in users:
    if 'name' in user and user.get('name') == bot_name:
        bot_id = user.get('id')

# Let's find the channel ID's and select the relevant ones:
channel_list = slack_client.api_call("channels.list")['channels']

def parse_slack_output(slack_rtm_output):
    """
    Check if the output from Slack came from a user as a text message
    """
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            print (output)
            if output and 'text' in output and 'user' in output:
                return (output['text'].strip().lower(), 
                		output['channel'], output['user'])
    return None, None, None

def handle_input(input_string, channel, user):
    """
    Handle the input and decide whether the bot reacts (and how) or not
    """
    # Clean the input string
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer=pickle.load(file)
    sequences = tokenizer.texts_to_sequences([input_string])
    input_clean = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    generic = False
    # If it's empty after cleaning, it's generic
    if len(input_clean) == 0: 
        print(len(input_clean))
        generic = True
    else:
        model=get_kmax_text_cnn(nb_words, EMBEDDING_DIM, embedding_matrix, MAX_SEQUENCE_LENGTH, out_size=6)
        model.load_weights('kmax_text_cnn1.h5')
        test_predicts = model.predict(input_clean, batch_size=1, verbose=1)
        test_predicts=test_predicts >= 0.5
        print(test_predicts)
        if True not in test_predicts: 
            generic = True
        else:
            response = ("Hey <@" + user + ">, your content might be offensive to some readers. Please consider rephrasing your message.")
            slack_client.api_call("chat.postEphemeral", channel = channel, text = response, user=user, as_user = True)

# Open the Slack RTM firehose:
if slack_client.rtm_connect():
    print("TalkSick connected and monitoring...")
    while True:
        command, channel, user = parse_slack_output(slack_client.rtm_read())
        if command and channel:
            handle_input(command, channel, user)
        time.sleep(0.5)
else:
    print("Connection failed.")