import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils, plot_model
from keras.callbacks import TensorBoard
np.random.seed(7)

# define the raw dataset
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#create mapping of characters to integers
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# perpare the dataset of input to output
seq_length = 3
dataX = []
dataY = []

for i in range(0, len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])

x = np.reshape(dataX, (len(dataX), seq_length, 1))

# normalize
x = x / float(len(alphabet))
y = np_utils.to_categorical(dataY)

# create the model
model = Sequential()
model.add(LSTM(32, input_shape=(x.shape[1], x.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=500, batch_size=1, verbose=1, callbacks=[TensorBoard(log_dir='tensorboard/test')])
scores = model.evaluate(x, y, verbose=0)
print('Model Accuracy: %.2f%%' % (scores[1]*100))
for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, '>', result)


