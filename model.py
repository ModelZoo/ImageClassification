from model_zoo.model import BaseModel
import tensorflow as tf


class FashionMnistModel(BaseModel):
    def __init__(self, config):
        super(FashionMnistModel, self).__init__(config)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                            kernel_initializer='random_uniform')
        self.pool1 = tf.keras.layers.MaxPool2D(padding='same')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.conv2 = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                            kernel_initializer='random_uniform')
        self.pool2 = tf.keras.layers.MaxPool2D(padding='same')
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.flatten1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
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
    
    def optimizer(self):
        return tf.train.AdamOptimizer(self.config.get('learning_rate'))
    
    def init(self):
        self.compile(optimizer=self.optimizer(), loss='categorical_crossentropy', metrics=['accuracy'])
