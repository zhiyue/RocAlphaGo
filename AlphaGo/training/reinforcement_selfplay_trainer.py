import os
import json
import argparse
from AlphaGo.go import GameState
from multiprocessing import Process, Queue, Value
from AlphaGo.models.policy_value import CNNPolicyValue
from AlphaGo.preprocessing.preprocessing_rollout import Preprocess as PreprocessRollout

DEFAULT_OPTIMIZER       = 'SGD'
DEFAULT_LEARNING_RATE   = .003
DEFAULT_BATCH_SIZE      = 16
DEFAULT_EPOCH_SIZE      = 10000
DEFAULT_TEST_AMOUNT     = 400
DEFAULT_TRAIN_EVERY     = 10000
DEFAULT_SIMULATIONS     = 1600
DEFAULT_RESIGN_TRESHOLD = 0.05
DEFAULT_ALLOW_RESIGN    = 0.9

# metdata file
FILE_METADATA = 'metadata_policy_value_reinforcement.json'
# hdf5 training file
FILE_METADATA = 'training_samples.hdf5'
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


class MCTS_Tree_Node():
    """
       Alphago zero style mcts node
    """

    def __init__( self, state, policy, value, parent=None ):
        """
           init new mcts tree node:
           - state is a GameState
           - policy is neural network policy prediction
           - value  is neural network value  prediction
           - parent is parent tree node
        """

        self.state    = state
        self.value    = value
        self.policy   = policy
        self.parent   = parent
        self.children = []
        self.visits   = 0

    def select_expand_move( self ):
        """
           select tree expansion move
           return treenode and move
        """

        # TODO implement this

        # expand childnode?
        if len( children ) > 0:
            return childen[ 0 ].select_expand_move()

        # expand this node
        return self, self.policy[ 0 ]

    def expand_and_update( self, state, policy, value ):
        """
           add new child node and update parent nodes 
        """

        # add new child node
        child = MCTS_Tree_Node( state, policy, value, parent=self )
        self.children.append( child )

        # update all parent nodes
        parent = self.parent
        while parent is not None:

            # update node

            # next parent
            parent = parent.parent


class Games_Generator( Process ):
    """
       
    """

    def __init__( self, id, queue_save, queue_requests, queue_predictions, game_count, feature_list, board_size, allow_resign, resign_threshold ):

        Process.__init__(self)
        self.id                = id
        self.queue_save        = queue_save
        self.game_count        = game_count
        self.board_size        = board_size
        self.allow_resign      = allow_resign
        self.preprocessor      = PreprocessRollout( feature_list, size=board_size )
        self.queue_requests    = queue_requests
        self.resign_threshold  = resign_threshold
        self.queue_predictions = queue_predictions

    def run(self):

        while True:

            training_positions = []
            state   = GameState( size = self.board_size )
            request = self.preprocessor.state_to_single_tensor( state )
            self.queue_requests.put( ( self.id, request ) )

            # blocks 
            policy, value = self.queue_predictions.get()

            root = MCTS_Tree_Node( state, policy, value )

            for i in range( 200 ):
                request = self.preprocessor.state_to_single_tensor( state )
                self.queue_requests.put( ( self.id, request ) )
                policy, value = self.queue_predictions.get()

            # untill game terminates
                # expand gametree x times

                # select best move, create training_position data,

            # add training_positions to queue_save
            # increment counter
            sgf_id = self.game_count.increment()
            # save sgf
            print 'no hi ' + str( sgf_id )


class Games_Saver( Process ):
    """
       save games in queue to ... (hdf5?? and sgf? -> or save to sgf in Games_Generator)
    """

    def __init__( self, id, queue_save ):

        Process.__init__(self)
        self.id = id
        self.queue_save = queue_save

    def run(self):

        while True:

            training_samples = self.queue_save.get()

            # store samples in hdf5 file
            print 'stored ' + str( len( training_samples ) ) + ' samples'


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


def train_and_save_model( out_directory, model, metadata  ):
    """
       train model, save and return file name
    """

    return save_model( out_directory, model, metadata['epoch_count'] )


def compare_strenght( current_network_weight_file, new_network_weight_file, amount ):
    """
       let both network play vs eachother for #amount games
       return winning ratio for new model 
    """

    return 0.5


def run_training( metadata, out_directory, verbose ):
    """
       
    """

    # metadata file location
    metadata['meta_file'] = os.path.join(out_directory, FILE_METADATA)

    # create network
    network = CNNPolicyValue.load_model(metadata["model_file"])

    if metadata['best_model'] is None:

        # save initial model
        metadata["best_model"] = save_model( out_directory, network.model, 0 )
        # get model board size
        metadata["board_size"] = network.model.input_shape[-1]
        # get model feature list
        metadata["feature_list"] = network.preprocessor.get_feature_list()
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

    # start hdf5 saver process
    saver = Games_Saver( 0, queue_save )
    saver.start()

    # start game generator worker process
    for i in range( metadata["batch_size"] ):

        worker = Games_Generator( i, queue_save, queue_requests, queue_predictions[ i ], game_count, metadata["feature_list"],
                                  metadata["board_size"], metadata["allow_resign"], metadata["resign_treshold"] )
        worker.start()

    while True:

        # start processing forward requests untill enough games have been played
        while game_count.get_value() < metadata['next_training_point']:

            metadata['game_count'] = game_count.get_value()
            requests   = []
            worker_ids = []

            # get as many request as possible
            while not queue_requests.empty():

                worker_id, request = queue_requests.get()
                requests.append( request )
                worker_ids.append( worker_id )
            
            print 'requests ' + str( len( requests ) )

            if len( requests ) > 0:

                # predict all requests
                predictions = network.forward( requests )

                # return prediction to corresponding worker
                for worker_id, policy, value in zip( worker_ids, predictions[ 0 ], predictions[ 1 ]  ):

                    # add policy and value prediction to worker queue
                    queue_predictions[ worker_id ].put( ( policy, value ) )


        # train and save new model
        new_model_file = train_and_save_model( out_directory, network.model, metadata )

        # test model strength vs old version
        ratio = compare_strenght( metadata['best_model'], new_model_file, metadata['test_amount'] )

        # check if new model beats previous model
        if ratio >= 0.55:

            metadata['best_model'] = new_model_file
            # load best model
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
        "next_training_point": args.train_every,
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
