import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy

def xor(hidden_units, init_hidden_bias):
  # data
  X = numpy.array([[-1., -1.], [1., 1.], [-1., 1.], [1., -1.]])
  Y = numpy.array([-1., -1., 1., 1.])
  # model
  model = Sequential()
  model.add(Dense(hidden_units,
                  input_shape=(2,),
                  activation='relu',
                  bias_initializer=keras.initializers.Constant(value=init_hidden_bias)))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(X, Y, epochs=3000, batch_size=4)


if __name__ == '__main__':
  # params = [(40, 0.0)]
  params = [(2, 0.0), (2, 10.0), (40, 0.0), (40, 10.0)]
  for hidden_units, init_hidden_bias in params:
    print('hidden_units:', hidden_units, 'init_hidden_bias:', init_hidden_bias) 
    xor(hidden_units, init_hidden_bias)

