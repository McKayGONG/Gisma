"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run GedGNN.")

    parser.add_argument("--epochs",
                        type=int,
                        default=1,
	                help="Number of training epochs. Default is 1.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
	                help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
	                help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=32,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--bottle-neck-neurons",
                        type=int,
                        default=16,
	                help="Bottle neck layer neurons. Default is 16.")

    parser.add_argument("--bottle-neck-neurons-2",
                        type=int,
                        default=8,
                        help="2nd bottle neck layer neurons. Default is 8.")

    parser.add_argument("--bottle-neck-neurons-3",
                        type=int,
                        default=4,
                        help="3rd bottle neck layer neurons. Default is 4.")

    parser.add_argument("--bins",
                        type=int,
                        default=16,
	                help="Similarity score bins. Default is 16.")

    parser.add_argument("--path",
                        dest="path",
                        action="store_true",
                        default=False,
                        help='Whether to generate edit path.'
                        )
    parser.add_argument("--GW",
                        dest="GW",
                        action="store_true",
                        default=False,
                        help='Whether to use GEDGW.'
                        )
    parser.add_argument("--hidden-dim",
                        type=int,
                        default=16,
                        help="the size of weight matrix in GedMatrixModule. Default is 16.")

    parser.add_argument("--histogram",
                        dest="histogram",
                        default=False,
                        help='Whether to use histogram.')

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.5,
	                help="Dropout probability. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")

    parser.add_argument("--demo",
                        dest="demo",
                        action="store_true",
                        default=False,
                        help='Generate just a few graph pairs for training and testing.')

    parser.add_argument("--gtmap",
                        dest="gtmap",
                        action="store_true",
                        default=False,
                        help='Whether to pack gt mapping')

    parser.add_argument("--finetune",
                        dest="finetune",
                        action="store_true",
                        default=False,
                        help='Whether to use finetune.')

    parser.add_argument("--prediction-analysis",
                        action="store_true",
                        default=False,
                        help='Whether to analyze the bias of prediction.')

    parser.add_argument("--postk",
                        type=int,
                        default=1,
                        help="Find k-best matching in the post-processing algorithm. Default is 1000.")

    parser.add_argument("--abs-path",
                        type=str,
                        default="",
                        # default='/apdcephfs/private_czpiao/workplace/gedgnn/',
                        help="the absolute path")

    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")

    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")

    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=0,
                        help="The number of epochs the final saved model has been trained.")

    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")

    parser.add_argument("--model-name",
                        type=str,
                        default='GEDGW',
                        help="model name, including [GPN, SimGNN, TaGSim, GedGNN, GEDGW, GEDIOT, GEDHOT].")

    # GEDGW-specific options
    parser.add_argument("--gw-alpha",
                        type=float,
                        default=1.0/3.0,
                        help="Alpha in fused Gromov-Wasserstein (0~1). Default is 1/3.")

    parser.add_argument("--gw-rescale",
                        type=float,
                        default=1.5,
                        help="Rescale factor applied to FGW distance to predict GED. Default is 1.5.")

    parser.add_argument("--gw-square",
                        dest="gw_square",
                        action="store_true",
                        default=True,
                        help='Pad the smaller graph to square cost (default: True).')

    parser.add_argument("--no-gw-square",
                        dest="gw_square",
                        action="store_false",
                        help='Disable square padding, use rectangular cost variant.')

    # Delta-only training mode (no TaGED required)
    parser.add_argument("--delta-only",
                        dest="delta_only",
                        action="store_true",
                        default=False,
                        help='Use only synthetic delta graph pairs for train/val/test; generate deltas for all graphs.')

    parser.add_argument("--graph-pair-mode",
                        type=str,
                        default='combine',
                        help="The way of generating graph pairs, including [normal, delta, combine].")

    parser.add_argument("--target-mode",
                        type=str,
                        default='linear',
                        help="The way of generating target, including [linear, exp].")

    parser.add_argument("--greedy",
                        dest="greedy",
                        action="store_true",
                        default=False,
                        help='Whether to use greedy matching matrix (Hungarian).')

    parser.add_argument("--num-delta-graphs",
                        type=int,
                        default=100,
                        help="The number of synthetic delta graph pairs for each graph.")

    parser.add_argument("--num-testing-graphs",
                        type=int,
                        default=100,
                        help="The number of testing graph pairs for each graph.")

    parser.add_argument("--loss-weight",
                        type=float,
                        default=1.0,
                        help="In GedGNN, the weight of value loss. Default is 1.0.")

    parser.add_argument("--parallel-loading",
                        dest="parallel_loading",
                        action="store_true",
                        default=True,
                        help="Enable parallel loading of graph files. Default is True.")
    
    parser.add_argument("--no-parallel-loading",
                        dest="parallel_loading", 
                        action="store_false",
                        help="Disable parallel loading (use serial loading).")
    
    parser.add_argument("--loading-workers",
                        type=int,
                        default=4,
                        help="Number of parallel workers for loading files. Default is 4.")



    # Similarity search parameters (unified interface)
    parser.add_argument("--similarity-search",
                        dest="similarity_search",
                        action="store_true",
                        default=False,
                        help='Enable similarity search mode')

    parser.add_argument("--batch-parallel",
                        dest="batch_parallel",
                        action="store_true",
                        default=False,
                        help='Enable batch-parallel mode: compute all distance pairs in parallel (only for GEDGW)')

    parser.add_argument("--no-summary",
                        dest="no_summary",
                        action="store_true",
                        default=False,
                        help='Skip automatic summary generation (only save per-query JSON files)')

    parser.add_argument("--method",
                        type=str,
                        choices=["GEDGW", "GEDIOT", "GEDHOT", "all"],
                        default="GEDIOT",
                        help='Method for similarity search: GEDGW (unsupervised), GEDIOT (neural network), GEDHOT (hybrid), all (compare all three)')

    parser.add_argument("--candidates-dir",
                        type=str,
                        default="candidates",
                        help="Directory containing candidates CSV files (default: 'candidates')")

    parser.add_argument("--tau-thresholds",
                        type=str,
                        default="2,4,6,8",
                        help="Comma-separated tau threshold values (default: '2,4,6,8')")

    parser.add_argument("--query-start",
                        type=int,
                        default=0,
                        help="Starting query ID for similarity search (default: 0)")

    parser.add_argument("--query-end",
                        type=int,
                        default=99,
                        help="Ending query ID for similarity search (default: 99)")

    parser.add_argument("--parallel-search",
                        dest="parallel_search",
                        action="store_true",
                        default=False,
                        help="Enable parallel processing for similarity search (default: False)")

    parser.add_argument("--parallel-workers",
                        type=int,
                        default=14,
                        help="Number of parallel workers for batch-parallel mode (default: 14)")

    # MAE test parameters
    parser.add_argument("--mae-test",
                        dest="mae_test",
                        action="store_true",
                        default=False,
                        help='Enable MAE test mode - test three methods against Gisma ground truth')

    parser.add_argument("--ged-results-file",
                        type=str,
                        default="",
                        help="Path to Gisma GED results file (format: id1,id2,ged)")

    parser.add_argument("--samples-per-ged",
                        type=str,
                        default="10",
                        help="Number of samples per GED value for MAE test. Can be: single number (e.g., '10') for all GED values, or comma-separated list (e.g., '5,10,15,20') for specific GED values (default: '10')")

    parser.add_argument("--ged-range-min",
                        type=int,
                        default=0,
                        help="Minimum GED value for MAE test (default: 0)")

    parser.add_argument("--ged-range-max",
                        type=int,
                        default=20,
                        help="Maximum GED value for MAE test (default: 20)")

    args = parser.parse_args()
    
    # Fix: Only train once if model_epoch_end is not explicitly set
    if args.model_epoch_end == 0:
        args.model_epoch_end = 1  # Only run the outer loop once
    
    return args
