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
       uses a convolutional neural network with a residual block part to evaluate the state of a game
       computes probability distribution over the next action and the win probability of the current player
    """

    def _select_moves_and_normalize(self, nn_output, moves, size):
        """
           helper function to normalize a distribution over the given list of moves
           and return a list of (move, prob) tuples
        """

        if len(moves) == 0:
            return []
        move_indices = [flatten_idx(m, size) for m in moves]
        # add pass move location
        move_indices.append( size * size )
        # get network activations at legal move locations
        distribution = nn_output[move_indices]
        distribution = distribution / distribution.sum()
        # add pass move value -> change to _PASS
        moves.append(None)

        return zip(moves, distribution)

    def eval_state(self, state, moves=None):
        """
           Given a GameState object, returns a tuple with alist of (action, probability) pairs
           according to the network outputs and win probability of current player

           If a list of moves is specified, only those moves are kept in the distribution
        """

        tensor = self.preprocessor.state_to_tensor(state)
        # run the tensor through the network
        network_output = self.forward(tensor)
        moves = moves or state.get_legal_moves()

        actions = self._select_moves_and_normalize(network_output[0][0], moves, state.get_size())

        return ( actions, network_output[1][0][0])

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
           - value_activation: value head output activation                               (default tanh)
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
            "value_activation": 'tanh',
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
            layer = add( [ layer, residual ] )
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
