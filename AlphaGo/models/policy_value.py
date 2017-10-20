from keras.models import Model
from keras.layers import *
from keras.layers.merge import add
from keras.layers.core import Activation, Flatten
from AlphaGo.util import flatten_idx
from AlphaGo.models.nn_util import Bias, NeuralNetBase, neuralnet
import numpy as np


@neuralnet
class CNNPolicyValue(NeuralNetBase):
    """
       uses a convolutional neural network to evaluate the state of the game
       and compute a probability distribution over the next action and win ratio for the currect position
    """

    @staticmethod
    def create_network(**kwargs):
        """
           construct a alphago zero style residual neural network.

           Keword Arguments:
           - board:            width of the go board to be processed                      (default 19)
           - input_dim:        depth of features to be processed by first layer           (no default)
           - activation:       type of activation used eg relu sigmoid                    (default relu)

             pre residual block convolution
           - conv_filter:      number of filters used on first convolution layer          (default 256)
           - conv_kernel:      kernel size used in first convolution layer                (default 3)   (Must be odd)

             residual block 
           - residual_depth:   number of residual blocks                                  (default 39)
           - residual_filter:  number of filters used on residual block convolution layer (default 256)
           - residual_kernel:  kernel size used in first residual block convolution layer (default 3)   (Must be odd)

             value head
           - value_size:       size of fully connected layer for value output             (default 256)
           - value_activation: value head output activation                               (default tahn)
        """

        defaults = {
            "board": 19,
            "activation": 'relu',
            "conv_filter": 256,
            "conv_kernel" : 3,
            "residual_depth" : 39,
            "residual_filter" : 256,
            "residual_kernel" : 3,
            "value_size": 256,
            "value_activation": 'tahn',
        }
        # copy defaults, but override with anything in kwargs
        params = defaults
        params.update(kwargs)

        # create input with theano ordering ( "channels_first" )
        inp = Input( shape=( params["input_dim"], params["board"], params["board"] ) )

        # create convolution layer
        layer = Conv2D( params["conv_filter"], ( params["conv_kernel"], params["conv_kernel"] ), data_format="channels_first", padding='same', name='conv1' )( inp )
        layer = BatchNormalization( name='conv1_bn')( layer )
        layer = Activation( params["activation"] )( layer )

        # create residual blocks
        for i in range( params["residual_depth"] ):

            # residual block comon name
            name = 'residual_block_' + str( i ) + '_'

            # first residual block convolution
            residual = Conv2D( params["residual_filter"], ( params["residual_kernel"], params["residual_kernel"] ), data_format="channels_first", padding='same', name=name + 'conv1' )( layer )
            residual = BatchNormalization( name=name + 'conv1_bn')( residual )
            residual = Activation( params["activation"] )( residual )

            # second residual block convolution
            residual = Conv2D( params["residual_filter"], ( params["residual_kernel"], params["residual_kernel"] ), data_format="channels_first", padding='same', name=name + 'conv2' )( residual )
            residual = BatchNormalization( name=name + 'conv2_bn')( residual )
            residual = Activation( params["activation"] )( residual )

            # add residual block input
            layer = add( [ x, residual ] )
            layer = Activation( params["activation"] )( layer )


        # create policy head
        policy = Conv2D( 2, ( 1, 1 ) )( layer )
        policy = BatchNormalization()( policy )
        policy = Activation( params["activation"] )( policy )
        policy = Flatten()( policy )
        # board * board for board locations, +1 for pass move
        policy = Dense( ( params["board"] * params["board"] ) + 1, activation='softmax', name='policy_output' )( policy )

        # create value head
        value = Conv2D(1, (1, 1) )( layer )
        value = BatchNormalization()( value )
        value = Activation( params["activation"] )( value )
        value = Flatten()( value )
        value = Dense( params["value_size"], activation=params["activation"] )( value )
        value = Dense( 1, activation=params["value_activation"], name='value_output' )( value )

        # create the network:
        network = Model( inp, [ policy, value ] )

        return network

