import numpy as np
from tensorflow import keras as k
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, \
  Flatten, Activation, RepeatVector, multiply, Permute, Lambda, Dropout, SimpleRNN
from readfile import load_data, embed_and_token

UNITS = 128 # number of hidden units in the lstm
REG = 0.0001
DROP = 0.01

tweets, labels = load_data()
data, labels, embedding_layer = embed_and_token(tweets, labels)
print(data.shape)
print(labels.shape)

sequence_input = Input(shape=(data.shape[1],), dtype='int32') # (Batch size,
embedded_sequences = embedding_layer(sequence_input)

x = LSTM(UNITS, return_sequences=True, kernel_regularizer=k.regularizers.l2(REG))(embedded_sequences)
x = LSTM(UNITS, return_sequences=True, kernel_regularizer=k.regularizers.l2(REG))(x)
# x = LSTM(UNITS, return_sequences=True, kernel_regularizer=k.regularizers.l2(REG))(x)
# x = LSTM(UNITS, return_sequences=True, kernel_regularizer=k.regularizers.l2(REG))(x)
x = LSTM(UNITS, kernel_regularizer=k.regularizers.l2(REG))(x)

x = Dense(UNITS, activation='relu')(x)
out = Dense(1, activation='sigmoid')(x)

model = k.models.Model(sequence_input, out)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

model.fit(data, labels, validation_split=0.2, epochs=20, batch_size=128)
