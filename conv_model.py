import numpy as np
from tensorflow import keras as k
from tensorflow.python.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Input, AveragePooling1D, Dropout, Concatenate
from tensorflow.python.keras import regularizers
from readfile import load_data, embed_and_token

REG = 0.0001
DROP = 0.1

tweets, labels = load_data()
data, labels, embedding_layer = embed_and_token(tweets, labels)
print(data.shape)
print(labels.shape)


sequence_input = Input(shape=(data.shape[1],), dtype='int32') # (Batch size,
embedded_sequences = embedding_layer(sequence_input)
embedded_sequences = Dropout(DROP)(embedded_sequences)
x = Conv1D(256, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(REG))(embedded_sequences)
a = MaxPooling1D(5, strides=1, padding='same')(x)
a = Dense(256, activation='relu')(a)
x = Dropout(DROP)(a)
x = Conv1D(256, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(REG))(x)
b = MaxPooling1D(5, strides=1, padding='same')(x)
b = Dense(256, activation='relu')(b)
x = Dropout(DROP)(b)
x = Conv1D(128, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(REG))(x)
x = MaxPooling1D(31, padding='same')(x)
x = Flatten()(x)
# a = Flatten()(a)
# b = Flatten()(b)
# x = Concatenate()([a, b, x])
x = Dense(128, activation='relu')(x)

out = Dense(1, activation='relu')(x)

opt = k.optimizers.Adam(lr=0.001)
model = k.models.Model(sequence_input, out)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print(model.summary())

model.fit(data, labels, validation_split=0.2, epochs=4, batch_size=512)