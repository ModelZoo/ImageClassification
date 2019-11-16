from tensorflow_core.python.keras.applications.vgg19 import VGG19

from model_zoo import Model
import tensorflow as tf


class VGG19Model(Model):
    """
    This model gets no good results, deprecated.
    """
    
    def __init__(self, **kwargs):
        """
        Init base model.
        """
        self.base_model = VGG19()
        super(VGG19Model, self).__init__(**kwargs)
    
    def inputs(self):
        """
        Define inputs.
        :return:
        """
        return self.base_model.input
    
    def outputs(self, inputs):
        """
        Define outputs.
        :param inputs:
        :return:
        """
        x = self.base_model.output
        return tf.keras.layers.Dense(10, activation='softmax')(x)
    
    def optimizer(self):
        """
        Build optimizer.
        :return:
        """
        return tf.keras.optimizers.Adam(lr=self.config.get('learning_rate'))
    
    def loss(self):
        """
        Define loss.
        :return:
        """
        return 'categorical_crossentropy'
    
    def metrics(self):
        """
        Define metrics.
        :return:
        """
        return ['accuracy']
