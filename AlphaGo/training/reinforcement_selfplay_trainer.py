import os
import json
import h5py
import argparse
import numpy as np
from AlphaGo.go import GameState
from AlphaGo.util import save_gamestate_to_sgf
from multiprocessing import Process, Queue, Value
from AlphaGo.models.policy_value import CNNPolicyValue
from AlphaGo.preprocessing.preprocessing_rollout import Preprocess as PreprocessRollout

import time

DEFAULT_OPTIMIZER       = 'SGD'
DEFAULT_LEARNING_RATE   = .003
DEFAULT_BATCH_SIZE      = 10    # combine multiple batches?? -> 2048
DEFAULT_EPOCH_SIZE      = 2048000
DEFAULT_TEST_AMOUNT     = 400
DEFAULT_TRAIN_EVERY     = 10
DEFAULT_SIMULATIONS     = 2
DEFAULT_RESIGN_TRESHOLD = 0.05  # should be automatically adjusted??
DEFAULT_ALLOW_RESIGN    = 0.9

# metdata file
FILE_METADATA = 'metadata_policy_value_reinforcement.json'
# hdf5 training file
FILE_HDF5 = 'training_samples.hdf5'
# weight folder
FOLDER_WEIGHT = os.path.join('policy_value_reinforcement_weights')
# folder sgf files
FOLDER_SGF = os.path.join('policy_value_reinforcement_sgf')

class Counter(object): 
    """
       Simple multiprocessing counter to keep track of game count
    """
   
    def __init__( self, value=0 ):

        self.count = Value( 'i', value )

    def increment(self):
        """
           increment count value and return value
        """
        with self.count.get_lock():

            self.count.value += 1
            return self.count.value

    def get_value(self):
        """
           get count value
        """
        with self.count.get_lock():

            return self.count.value

class HDF5_handler():
    """
       Simple Hdf5 file handler
    """

    def __init__( self, database ):

        self.idx           = len( database['action_value'] )
        # empty database has lenght 1
        if self.idx == 1:
            self.idx = 0

        self.database      = database
        self.states        = database['states']
        self.value         = database['action_value']
        self.policy        = database['action_policy']

    @staticmethod
    def create_hdf5_file( file_location, feature_list, board_size, depth ):
        """
           create hdf5 file
        """

        # todo check compression speed options
        print 'create'

        # create hdf5 file
        database = h5py.File(file_location, 'w') # w Create file, truncate if exists

        try:

            # create states ( network inputs )
            database.create_dataset(
                                     'states',
                                     dtype=np.uint8,
                                     shape=(1, depth, board_size, board_size),
                                     maxshape=(None, depth, board_size, board_size), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, depth, board_size, board_size),
                                     compression="lzf"
                                    )

            # create action_value ( network value output ) -> do we want to train with multiple komi values? https://arxiv.org/pdf/1705.10701.pdf
            # TODO check if this needs to be a float value or are these only 1 -1????
            database.create_dataset(
                                     'action_value',
                                     dtype=np.uint8,
                                     shape=(1, 1),
                                     maxshape=(None, 1), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, 1),
                                     compression="lzf"
                                    )

            # create action_policy ( network policy output )
            database.create_dataset(
                                     'action_policy',
                                     dtype=np.float16,
                                     shape=(1, ( board_size * board_size ) + 1 ),       # +1 for pass move
                                     maxshape=(None, ( board_size * board_size ) + 1 ), # 'None' dimension allows it to grow arbitrarily
                                     chunks=(1, ( board_size * board_size ) + 1 ),
                                     compression="lzf"
                                    )

            # add features
            database['features'] = np.string_(','.join(feature_list))


        except Exception as e:
            raise e

        # close file
        database.close()

    def add_samples( self, state_samples, value_samples, policy_samples ):
        """
           add samples to hdf5 file
        """

        # calculate new database size
        size_new = self.idx + len( value_samples )

        try:

            # resize databases
            self.value.resize(  size_new, axis=0 )
            self.states.resize( size_new, axis=0 )
            self.policy.resize( size_new, axis=0 )

            # add samples
            self.value[self.idx:]  = value_samples
            self.states[self.idx:] = state_samples
            self.policy[self.idx:] = policy_samples

            self.idx = size_new

        except Exception as e:
            raise e

    def get_random_samples( self, max_range, amount ):
        """
           return #amount samples in #max_range from the end
        """

        # TODO
        # aply rotations here???
        print 'return ' + str( amount ) + ' samples'


class MCTS_Tree_Node():
    """
       Alphago zero style mcts node
    """

    def __init__( self, state, policy, value, nn_input, parent=None ):
        """
           init new mcts tree node:
           - state is a GameState
           - policy is neural network policy prediction
           - value  is neural network value  prediction
           - nn_input is neural network input ( prevent double preprocessing )
           - parent is parent tree node
        """

        self.state    = state
        self.value    = value
        self.visits   = 0
        self.policy   = policy
        self.parent   = parent
        self.children = {}
        self.nn_input = nn_input

    def select_expand_move( self ):
        """
           select tree expansion move
           return treenode and move
        """

        # TODO implement this
        # select expand move
        # move  = 0
        # value = 0
        # for i in range( len( self.policy ) ):
        #     if policy[ i ] > value:
        #         move = i
        # if move in self.children:
        #     return self.children[ move ].select_expand_move()
        # else
        #     return self, move

        # expand childnode?
        if len( self.children ) > 0:
            return childen[ 0 ].select_expand_move()

        # expand this node
        return self, self.policy[ 0 ]

    def expand_and_update( self, move, state, policy, value, nn_input ):
        """
           add new child node and update parent nodes 
        """

        # add new child node
        child = MCTS_Tree_Node( state, policy, value, nn_input, parent=self )
        self.children.append( child )

        # update all parent nodes
        parent = self.parent
        while parent is not None:

            # update node

            # next parent
            parent = parent.parent

    def get_best_move( self ):
        """
           return best child node
        """

        # loop over all childer and select best

        return self


class Games_Generator( Process ):
    """
       
    """

    def __init__( self, id, queue_save, queue_requests, queue_predictions, game_count, feature_list, board_size, allow_resign, resign_threshold, simulations, out_directory ):

        Process.__init__(self)
        self.id                = id
        self.queue_save        = queue_save
        self.game_count        = game_count
        self.board_size        = board_size
        self.simulations       = simulations
        self.allow_resign      = allow_resign
        self.preprocessor      = PreprocessRollout( feature_list, size=board_size )
        self.out_directory     = out_directory
        self.queue_requests    = queue_requests
        self.resign_threshold  = resign_threshold
        self.queue_predictions = queue_predictions

    def run(self):

        while True:

            # lists to keep track of training samples
            training_state = []
            training_value = []
            training_policy = []

            # new game
            state   = GameState( size = self.board_size )
            # generate request
            # request = self.preprocessor.state_to_single_tensor( state )
            request = np.zeros( ( 17, 9, 9 ) )
            # request network prediction
            self.queue_requests.put( ( self.id, request ) )

            # get network prediction ( blocks )
            policy, value = self.queue_predictions.get()
            # new MCTS tree
            root = MCTS_Tree_Node( state, policy, value, request )

            # allow resign?

            # play game until resign or termination
            # while True:
            # simulate 10 move game
            for i in range( 30 ):

                # run mcts exploration
                for _ in range( self.simulations ):

                    node, move = root.select_expand_move()

                    # get new boardstate with move
                    # state = node.state.do_move( move )

                    # get nn prediction
                    #request = self.preprocessor.state_to_single_tensor( node.state )
                    request = np.zeros( ( 17, 9, 9 ) )

                    # TODO random rotation

                    self.queue_requests.put( ( self.id, request ) )
                    policy, value = self.queue_predictions.get()

                    # TODO rotate policy back

                    # expand node
                    # node.expand_and_update( move, state, policy, value, request )

                # generate training samples

                # training state sample is preprocessed state ( nn input )
                # TODO resusing the request send via Queue seems to create problems, find out why
                training_state.append( root.nn_input )

                # training policy sample is board_size + 1 array with 
                # child.visits / total.visits ( creating a heat map of mcts visits )
                training_policy.append( np.zeros( ( self.board_size * self.board_size + 1 ) ) )

                # add 1 for white move, -1 for black move
                # after all games are over we can multiply with -1 to get correct values
                training_value.append( [ 1 ] )

                # select best move and set new root or resign
                # check pass and resign
                # root = root.get_best_move()

            # determine winner and update training_value if needed

            # add training_positions to queue_save
            # correct order should be: state, value, policy
            self.queue_save.put( [ training_state, training_value, training_policy ] )

            # increment game counter
            sgf_id = self.game_count.increment()

            # save game to sgf
            file_name = "game.{version:08d}.sgf".format(version=sgf_id)
            save_file = os.path.join(self.out_directory, FOLDER_SGF)
            save_gamestate_to_sgf(root.state, save_file, file_name, result='black', size=self.board_size)


class Training_Samples_Saver( Process ):
    """
       save games in queue to hdf5
    """

    def __init__( self, id, queue_save, hdf5_file ):

        Process.__init__(self)
        self.id = id
        self.hdf5_file  = hdf5_file
        self.queue_save = queue_save

    def run(self):

        # open hdf5 file
        with h5py.File( self.hdf5_file, 'r+') as f:

            # create hdf5 handler
            database = HDF5_handler( f )

            while True:

                # correct order should be: state, value, policy
                # or None as stopping signal
                training_samples = self.queue_save.get() # blocking

                # check if process should stop
                if training_samples is None:

                    break

                # correct order should be: state, value, policy
                # store samples in hdf5 file
                database.add_samples( training_samples[0], training_samples[1], training_samples[2] )


def save_metadata( metadata ):
    """
       Save metadata
    """

    # update metadata file        
    with open( metadata['meta_file'], "w" ) as f:

        json.dump(metadata, f, indent=2)


def save_model( out_directory, model, version ):
    """
       Save network model weights to file and return file name
    """

    file_name = "weights.{version:05d}.hdf5".format(version=version)
    save_file = os.path.join(out_directory, FOLDER_WEIGHT, file_name)
    model.save(save_file)

    return file_name


def train_and_save_model( out_directory, model_file, weight_file, version ):
    """
       train model, save and return file name
    """

    # load newest model
    network = CNNPolicyValue.load_model( model_file )
    network.model.load_weights(os.path.join(out_directory, FOLDER_WEIGHT, weight_file))

    # train model
    # TODO

    # save model
    file_name = save_model( out_directory, network.model, version )

    # results
    result = {
             'accuracy': 0.5,
             'loss':1.22,
             'version': version,
             'file': file_name
             }

    return result


def compare_strenght( model_file, current_network_weight_file, new_network_weight_file, amount ):
    """
       let both network play vs eachother for #amount games
       return winning ratio for new model 
    """

    # TODO

    return 0.5


def run_training( metadata, out_directory, verbose ):
    """
       Run training pipeline
    """

    # metadata file location
    metadata['meta_file'] = os.path.join(out_directory, FILE_METADATA)

    # hdf5 file location
    hdf5_file = os.path.join(out_directory, FILE_HDF5)

    # create network
    network = CNNPolicyValue.load_model(metadata["model_file"])

    # 
    if metadata['best_model'] is None:

        # save initial model
        metadata["best_model"] = save_model( out_directory, network.model, 0 )
        metadata["newest_model"] = metadata["best_model"]
        # get model board size
        metadata["board_size"] = network.model.input_shape[-1]
        # get model feature list
        metadata["feature_list"] = network.preprocessor.get_feature_list()
        # create hdf5 file
        HDF5_handler.create_hdf5_file( hdf5_file, network.preprocessor.get_feature_list(), metadata["board_size"], network.preprocessor.get_output_dimension() )
    else:

        # load best model
        network.model.load_weights(os.path.join(out_directory, FOLDER_WEIGHT, metadata['best_model']))

    # queue with samples to save to hdf5
    queue_save        = Queue()
    # queue with positions to network.forward
    queue_requests    = Queue()
    # list with queue for each worker
    queue_predictions = [ Queue() for _ in range( metadata["batch_size"] )]
    # game counter 
    game_count = Counter( value=metadata['game_count'] )

    # start game generator worker process
    for i in range( metadata["batch_size"] ):

        worker = Games_Generator( i, queue_save, queue_requests, queue_predictions[ i ], game_count, metadata["feature_list"],
                                  metadata["board_size"], metadata["allow_resign"], metadata["resign_treshold"], metadata["simulations"],
                                  out_directory )
        worker.start()

    ###############################################
    ################################ generate games
    while True:

        # start hdf5 saver process
        saver = Training_Samples_Saver( 0, queue_save, hdf5_file )
        saver.start()

        if verbose:
            print 'Generating self play data'

        processed = 0

        # start processing forward requests untill enough games have been played
        while game_count.get_value() < metadata['next_training_point']:

            #print str( game_count.get_value() ) + ' - ' + str( processed )

            requests   = []
            worker_ids = []

            # get as many request as possible
            while not queue_requests.empty():

                worker_id, request = queue_requests.get()
                requests.append( request )
                worker_ids.append( worker_id )

            if len( requests ) > 0:

                processed += len( requests )

                # predict all requests
                predictions = network.forward( requests )

                # return prediction to corresponding worker
                for worker_id, policy, value in zip( worker_ids, predictions[ 0 ], predictions[ 1 ]  ):

                    # add policy and value prediction to worker queue
                    queue_predictions[ worker_id ].put( ( policy, value ) )

        # stop hdf5 save thread just to be sure it will not interfere with training
        queue_save.put(None)
        # wait for process to stop
        saver.join()

        # update game count
        metadata['game_count'] = game_count.get_value()

        if verbose:
            print 'Training new model'

        # train and save new model
        result = train_and_save_model( out_directory, metadata["model_file"], metadata["newest_model"], metadata['epoch_count'] )
        metadata["newest_model"] = result['file']

        if verbose:
            print 'Testing new model strength'

        # test model strength vs old version
        ratio = compare_strenght( metadata["model_file"], metadata['best_model'], result['file'], metadata['test_amount'] )
        result['opponent'] = metadata['best_model']
        result['winratio'] = ratio
        result['comparegamecount'] = metadata['test_amount']
        result['currentgamecount'] = metadata['game_count']
        # add new model results
        metadata['model_verions'].append( result )

        # check if new model beats previous model with margin
        if ratio >= 0.55:

            metadata['best_model'] = result['file']
            # load new best model
            network.model.load_weights(os.path.join(out_directory, FOLDER_WEIGHT, metadata['best_model']))

        # update for next training point
        metadata['epoch_count'] += 1
        metadata['next_training_point'] += metadata['train_every']

        # update metadata file        
        save_metadata( metadata )


def start_training(args):
    """
       create metadata, check argument settings and start training
    """

    # create metadata
    metadata = {
        "next_training_point": args.train_every * 2, # first time we want to get enough games before training
        "resign_treshold": args.resign_treshold,
        "learning_rate": args.learning_rate,
        "model_verions": [],
        "allow_resign": args.allow_resign,
        "train_every": args.train_every,
        "test_amount": args.test_amount,
        "simulations": args.simulations,
        "epoch_size": args.epoch_size,
        "batch_size": args.minibatch,
        "best_model": args.weights,
        "optimizer": args.optimizer,
        "model_file": args.model,
        "game_count": 0,
        "epoch_count": 1
    }

    # check if optimizer is supported
    if metadata['optimizer'] != 'SGD':

        raise ValueError("Optimizer is not supported!")

    # create all directories
    # main folder
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    # create weights file folder
    weight_folder = os.path.join(args.out_directory, FOLDER_WEIGHT)
    if not os.path.exists(weight_folder):
        os.makedirs(weight_folder)

    # create sgf save file folder
    sgf_folder = os.path.join(args.out_directory, FOLDER_SGF)
    if not os.path.exists(sgf_folder):
        os.makedirs(sgf_folder)

    # save metadata to file
    meta_file = os.path.join(args.out_directory, FILE_METADATA)
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # start training
    run_training( metadata, args.out_directory, args.verbose )


def resume_training(args):
    """
       Read metadata file and resume training
    """

    # metadata json file location
    meta_file = os.path.join(args.out_directory, FILE_METADATA)

    # load data from json file
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            metadata = json.load(f)
    else:
        raise ValueError("Metadata file not found!")

    # check if we need to train or validate new model

    # start training
    run_training( metadata, args.out_directory, args.verbose )


def handle_arguments( cmd_line_args=None ):
    """
       argument parser for training and resume training
    """

    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    # subparser is always first argument
    subparsers = parser.add_subparsers(help='sub-command help')

    ########################################
    ############## sub parser start training
    train = subparsers.add_parser('train', help='Start or resume supervised training on a policy network.')  # noqa: E501

    ####################
    # required arguments
    train.add_argument("out_directory", help="directory where metadata and weights will be saved")  # noqa: E501
    train.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicyValue.save_model())")  # noqa: E501

    ####################
    # optional arguments
    train.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to start training with. Default: None", default=None)  # noqa: E501
    train.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    train.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: " + str(DEFAULT_BATCH_SIZE), type=int, default=DEFAULT_BATCH_SIZE)  # noqa: E501
    train.add_argument("--epoch-size", "-E", help="Amount of batches per epoch. Default: " + str(DEFAULT_EPOCH_SIZE), type=int, default=DEFAULT_EPOCH_SIZE)  # noqa: E501
    train.add_argument("--test-amount", help="Amount of games to play to determine best model. Default: " + str(DEFAULT_TEST_AMOUNT), type=int, default=DEFAULT_TEST_AMOUNT)  # noqa: E501
    train.add_argument("--train-every", "-T", help="Train new model after this many games. Default: " + str(DEFAULT_TRAIN_EVERY), type=int, default=DEFAULT_TRAIN_EVERY)  # noqa: E501
    train.add_argument("--optimizer", "-O", help="Used optimizer. (SGD) Default: " + DEFAULT_OPTIMIZER, type=str, default=DEFAULT_OPTIMIZER)  # noqa: E501
    train.add_argument("--simulations", "-s", help="Amount of MCTS simulations per move. Default: " + str(DEFAULT_SIMULATIONS), type=int, default=DEFAULT_SIMULATIONS)  # noqa: E501
    train.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: " + str(DEFAULT_LEARNING_RATE), type=float, default=DEFAULT_LEARNING_RATE)  # noqa: E501
    train.add_argument("--resign-treshold", help="Resign treshold. Default: " + str(DEFAULT_RESIGN_TRESHOLD), type=float, default=DEFAULT_RESIGN_TRESHOLD)  # noqa: E501
    train.add_argument("--allow-resign", help="Percentage of games allowed to resign game. Default: " + str(DEFAULT_ALLOW_RESIGN), type=float, default=DEFAULT_ALLOW_RESIGN)  # noqa: E501
    # TODO add any variables we need that might vary
    #L2 regulizer

    # function to call when start training
    train.set_defaults(func=start_training)

    ########################################
    ############# sub parser resume training
    resume = subparsers.add_parser('resume', help='Resume supervised training on a policy network. (Settings are loaded from savefile.)')  # noqa: E501

    ####################
    # required arguments
    resume.add_argument("out_directory", help="directory where metadata and weight files where stored during previous session.")  # noqa: E501

    ####################
    # optional arguments
    resume.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    # function to call when resume training
    resume.set_defaults(func=resume_training)

    # show help or parse arguments
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # execute function (train or resume)
    args.func(args)


if __name__ == '__main__':

    handle_arguments()
