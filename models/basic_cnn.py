from model_zoo.model import BaseModel
import tensorflow as tf


class BasicCNNModel(BaseModel):
    """
    Basic CNN Model
    """

    def __init__(self, config):
        """
        init layers
        :param config:
        """
        super(BasicCNNModel, self).__init__(config)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                            kernel_initializer='random_uniform')
        self.pool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.conv2 = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                            kernel_initializer='random_uniform')
        self.pool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False, mask=None):
        """
        build model
        :param inputs: inputs image
        :param training:
        :param mask:
        :return:
        """
        o = self.bn1(inputs)
        o = self.conv1(o)
        o = self.pool1(o)
        o = self.dropout1(o) if training else o
        o = self.conv2(o)
        o = self.pool2(o)
        o = self.dropout2(o) if training else o
        o = self.flatten1(o)
        o = self.dense1(o)
        o = self.dense2(o)
        return o

    def get_optimizer(self):
        """
        build optimizer
        :return:
        """
        return tf.keras.optimizers.Adam(lr=self.config.get('learning_rate'))

    def get_loss(self):
        """
        define loss
        :return:
        """
        return 'categorical_crossentropy'

    def get_metrics(self):
        """
        define metrics
        :return:
        """
        return ['accuracy']
