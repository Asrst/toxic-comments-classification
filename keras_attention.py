from keras.models import Model
from keras import optimizers, callbacks, regularizers
from keras.layers import Input, Dense, Embedding, concatenate, Layer
from keras.callbacks import Callback

# keras attention layer
CONTEXT_DIM = 100

class Attention(Layer):

    def __init__(self, regularizer=regularizers.l2(1e-10), **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], CONTEXT_DIM),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b',
                                 shape=(CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u',
                                 shape=(CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)        
        super(Attention, self).build(input_shape)

    @staticmethod
    def softmax(x, dim):
        """Computes softmax along a specified dim. Keras currently lacks this feature.
        """
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.nn.softmax(x, dim)
        else:
            raise ValueError("Backend '{}' not supported".format(K.backend()))

    def call(self, x, mask=None):
        
        ut = K.tanh(K.bias_add(K.dot(x, self.W), self.b)) * self.u

        # Collapse `attention_dims` to 1. This indicates the weight for each time_step.
        ut = K.sum(ut, axis=-1, keepdims=True)

        # Convert those weights into a distribution but along time axis.
        # i.e., sum of alphas along `time_steps` axis should be 1.
        self.at = self.softmax(ut, dim=1)
        if mask is not None:
            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())

        # Weighted sum along `time_steps` axis.
        return K.sum(x * self.at, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = {}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None