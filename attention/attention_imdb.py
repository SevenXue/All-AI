from keras.preprocessing import sequence
from keras.datasets import imdb
from attention_keras import *

max_features = 20000
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test_shape', x_test.shape)

from keras.models import Model
from keras.layers import *

s_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(s_inputs)
# embeddings = Posi
out_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
out_seq = GlobalAveragePooling1D()(out_seq)
out_seq = Dropout(0.5)(out_seq)
outputs = Dense(1, activation='sigmoid')(out_seq)

model = Model(inputs=s_inputs, outputs=outputs)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))