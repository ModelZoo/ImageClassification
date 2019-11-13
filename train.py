import tensorflow as tf
from model_zoo.trainer import BaseTrainer
from model_zoo import datasets
from model_zoo import flags

flags.DEFINE_integer('epochs', 100)
flags.DEFINE_float('learning_rate', 0.01)
flags.DEFINE_string('model_class_name', 'BasicCNNModel')
flags.DEFINE_string('checkpoint_name', 'model.ckpt')


class Trainer(BaseTrainer):
    """
    Train Image Classification Model.
    """

    def prepare_data(self):
        """
        Prepare fashion mnist data.
        :return:
        """
        (x_train, y_train), (_, _) = datasets.fashion_mnist.load_data()
        x_train = x_train.reshape((-1, 28, 28, 1))
        x_train, y_train = x_train.astype('float16') / 255.0, \
                           tf.keras.utils.to_categorical(y_train.astype('float16'), 10)
        (x_train, x_eval) = x_train[5000:], x_train[:5000]
        (y_train, y_eval) = y_train[5000:], y_train[:5000]
        train_data, eval_data = (x_train, y_train), (x_eval, y_eval)
        return train_data, eval_data


if __name__ == '__main__':
    Trainer().run()
