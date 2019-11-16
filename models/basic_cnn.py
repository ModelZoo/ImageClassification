from model_zoo import Model
import tensorflow as tf


class BasicCNNModel(Model):
    """
    Basic CNN Model
    """
    
    def inputs(self):
        """
        Define inputs.
        :return:
        """
        return tf.keras.Input(shape=(28, 28, 1))
    
    def outputs(self, inputs):
        """
        Define outputs.
        """
        x = tf.keras.layers.BatchNormalization()(inputs)
        x = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                   kernel_initializer='random_uniform')(x)
        x = tf.keras.layers.MaxPool2D(padding='same')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Conv2D(32, (2, 2), padding='same', activation='relu',
                                   kernel_initializer='random_uniform')(x)
        x = tf.keras.layers.MaxPool2D(padding='same')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='random_uniform')(x)
        return tf.keras.layers.Dense(10, activation='softmax')(x)
    
    def optimizer(self):
        """
        build optimizer.
        :return:
        """
        return tf.keras.optimizers.Adam(lr=self.config.get('learning_rate'))
    
    def loss(self):
        """
        define loss.
        :return:
        """
        return 'categorical_crossentropy'
    
    def metrics(self):
        """
        define metrics.
        :return:
        """
        return ['accuracy']
