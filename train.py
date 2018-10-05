from model import FashionMnistModel
import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from tensorflow.python.keras.datasets import fashion_mnist

tf.flags.DEFINE_integer('epochs', 20, 'Max epochs')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')


class Trainer(BaseTrainer):
    
    def __init__(self):
        BaseTrainer.__init__(self)
        self.model_class = FashionMnistModel
    
    def prepare_data(self):
        (x_train, y_train), (x_eval, y_eval) = fashion_mnist.load_data()
        x_train, x_eval = x_train.reshape((-1, 28, 28, 1)), x_eval.reshape((-1, 28, 28, 1))
        x_train, x_eval = x_train.astype('float32') / 255.0, x_eval.astype('float32') / 255.0
        y_train, y_eval = y_train.astype('float32'), y_eval.astype('float32')
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data
        

if __name__ == '__main__':
    Trainer().run()
