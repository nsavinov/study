# dependencies:
# conda create --name split_plus
# conda activate split_plus
# conda install tensorflow-gpu cudatoolkit=9.0
# python -c 'import tensorflow as tf; print(tf.__version__)'
import tensorflow as tf
mnist = tf.keras.datasets.mnist


def get_split_plus(units, alpha):
    @tf.custom_gradient
    def split_plus(xy):
        get_first = tf.keras.layers.Lambda(lambda arg: arg[:, :units])
        get_second = tf.keras.layers.Lambda(lambda arg: arg[:, units:])
        x = get_first(xy)
        y = get_second(xy)
        def grad(dout):
            dx = dout + alpha * x
            dy = dout - alpha * x
            return tf.keras.layers.Concatenate()([dx, dy])
        return x + y, grad
    return split_plus


def split_dense(units, alpha, kernel_regularizer, input):
  wx = tf.keras.layers.Dense(units, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer))(input)
  b = tf.keras.layers.Dense(units, kernel_constraint=tf.keras.constraints.max_norm(.0))(input)
  wxb = tf.keras.layers.Lambda(get_split_plus(units, alpha))(tf.keras.layers.Concatenate()([wx, b]))
  return wxb


def run_experiment(shift, alpha, kernel_regularizer):
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train += shift
    inputs = tf.keras.layers.Input(shape=(28, 28,))
    flatten = tf.keras.layers.Flatten()(inputs)
    dense1 = split_dense(512, alpha, kernel_regularizer, flatten)
    dense1 = tf.keras.layers.ReLU()(dense1)
    outputs = split_dense(10, alpha, kernel_regularizer, dense1)
    outputs = tf.keras.layers.Softmax()(outputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    kernel_regularizer = 0.0
    for shift in [0.0, 10.0]:
        for alpha in [0.0, 0.001]:
            print('Shift:', shift, 'Alpha:', alpha, 'Kernel regularizer:', kernel_regularizer)
            run_experiment(shift, alpha, 0.0)
    kernel_regularizer = 0.1
    shift = 10.0
    alpha = 0.0
    print('Shift:', shift, 'Alpha:', alpha, 'Kernel regularizer:', kernel_regularizer)
    run_experiment(shift, alpha, kernel_regularizer)
