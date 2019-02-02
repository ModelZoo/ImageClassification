import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from tensorflow.python.keras.datasets import fashion_mnist

tf.flags.DEFINE_integer('epochs', 100, 'Max epochs')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.flags.DEFINE_string('model_class', 'VGG19Model', help='Model class name')


class Trainer(BaseTrainer):
    
    def prepare_data(self):
        (x_train, y_train), (_, _) = fashion_mnist.load_data()
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_train, y_train = x_train.astype('float32') / 255.0, \
                           tf.keras.utils.to_categorical(y_train.astype('float32'), 10)
        (x_train, x_eval) = x_train[5000:], x_train[:5000]
        (y_train, y_eval) = y_train[5000:], y_train[:5000]
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
