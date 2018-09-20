#TODO set up the embedding matrix so it isn't recomputed on every run
import numpy as np
from tensorflow import keras as k
from tensorflow.python.keras.layers import Bidirectional, LSTM, Dense, Input, TimeDistributed, \
  Flatten, Activation, RepeatVector, multiply, Permute, Lambda, Dropout
from readfile import load_data, embed_and_token

UNITS = 256 # number of hidden units in the lstm
REG = 0.01
DROP = 0.01

tweets, labels = load_data()
data, labels, embedding_layer = embed_and_token(tweets, labels)
print(data.shape)
print(labels.shape)

sequence_input = Input(shape=(data.shape[1],), dtype='int32') # (Batch size,
embedded_sequences = embedding_layer(sequence_input)
x = Bidirectional(LSTM(UNITS, return_sequences=True, dropout=DROP, activity_regularizer=k.regularizers.l2(REG)), merge_mode='concat')(embedded_sequences) # (batch_size, timesteps, units)
a = TimeDistributed(Dense(UNITS, activity_regularizer=k.regularizers.l2(REG)))(x)
attention = TimeDistributed(Dense(1, activation='tanh', name='timeDense'))(a) # (batch_size, timesteps, 1)
attention = Flatten()(attention) # (batch size, timesteps)
attention = Activation('softmax')(attention) # (batch, timesteps)
attention = RepeatVector(UNITS*2)(attention) # (batch, units, timesteps)
attention = Permute([2,1])(attention) #(batch, timesteps, units)
rejoined = multiply([x, attention])
# rejoined = k.backend.sum(rejoined, axis=-2 , keepdims=False)(rejoined)
# x = LSTM(UNITS, return_sequences=True, dropout=DROP, activity_regularizer=k.regularizers.l2(REG))(rejoined) # (batch_size, timesteps, units)
# x = TimeDistributed(Dense(UNITS, activation='relu', activity_regularizer=k.regularizers.l2(REG)))(x)
# attention = TimeDistributed(Dense(1, activation='tanh', name='timeDense'))(x) # (batch_size, timesteps, 1)
# attention = Flatten()(attention) # (batch size, timesteps)
# attention = Activation('softmax')(attention) # (batch, timesteps)
# attention = RepeatVector(UNITS)(attention) # (batch, units, timesteps)
# attention = Permute([2,1])(attention) #(batch, timesteps, units)
# rejoined = multiply([x, attention])

interm = LSTM(UNITS, activity_regularizer=k.regularizers.l2(REG), dropout=DROP)(rejoined)
interm = Dense(UNITS, activation='relu', kernel_regularizer=k.regularizers.l2(REG), bias_regularizer=k.regularizers.l2(REG))(interm)



# interm = LSTM(UNITS, dropout=DROP, activity_regularizer=k.regularizers.l2(REG))(rejoined)

out = Dense(1, activation='sigmoid', name='finaldense')(interm)

opt = k.optimizers.Adam(lr=0.001)
model = k.models.Model(sequence_input, out)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())

model.fit(data, labels, validation_split=0.2, epochs=20, batch_size=512)