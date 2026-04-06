# Completely fix OpenMP conflict
import os
import sys

# ========== Fix "Too many open files" issue ==========
# Must be set before all other imports
import multiprocessing
import torch
import platform

# 1. Key: use file_system to share tensors, avoids FD explosion
torch.multiprocessing.set_sharing_strategy("file_system")
print("PyTorch tensor sharing: file_system (prevents FD explosion)")

# 2. Use spawn mode (recommended by PyTorch, avoids fork deadlock issues)
try:
    multiprocessing.set_start_method('spawn', force=True)
    print("Multiprocessing mode: spawn (PyTorch-safe, avoids fork deadlocks)")
except RuntimeError:
    print(f"Multiprocessing mode: {multiprocessing.get_start_method()}")

# Set environment variables before any imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Pre-suppress OpenMP library loading
try:
    import ctypes
    # Try to load potentially conflicting OpenMP libraries and set to ignore
    try:
        libiomp = ctypes.cdll.LoadLibrary('libiomp5md.dll')
    except:
        try:
            libiomp = ctypes.cdll.LoadLibrary('mkl_rt.dll')
        except:
            pass  # OpenMP libraries not found or already loaded
except Exception as e:
    print(f"OpenMP setup warning: {e}")

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import numpy as np
import torch
import sys
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    seed_torch(1)
    args = parameter_parser()

    # Unified similarity search interface
    if args.similarity_search:
        print(f"=== Starting Similarity Search (Method: {args.method}) ===")

        # Parse tau thresholds
        try:
            tau_list = [float(t.strip()) for t in args.tau_thresholds.split(',')]
        except:
            print("Error: Invalid tau-thresholds format. Expected comma-separated values like '2,4,6,8'")
            sys.exit(1)

        # Validate required parameters
        if not args.candidates_dir:
            print("Error: --candidates-dir is required for similarity search")
            sys.exit(1)

        # Set model_name based on method
        if args.method in ["GEDIOT", "GEDHOT", "all"]:
            if not hasattr(args, 'model_name') or args.model_name == "GEDGW":
                args.model_name = "GEDIOT"
                print(f"Setting model_name to GEDIOT for method '{args.method}'")
        elif args.method == "GEDGW":
            args.model_name = "GEDGW"

        tab_printer(args)
        trainer = Trainer(args)

        # Load model if required
        if args.method in ["GEDIOT", "GEDHOT", "all"]:
            if args.model_epoch_start > 0:
                print(f"Loading GEDIOT model from epoch {args.model_epoch_start}")
                trainer.load(args.model_epoch_start)
            else:
                print("Error: No model epoch specified. Please use --model-epoch-start to specify the epoch.")
                sys.exit(1)

        # Call appropriate search method
        if args.method in ["GEDIOT", "GEDGW", "GEDHOT", "all"]:
            # Check if batch-parallel mode is requested
            if args.batch_parallel:
                print(f"=== Using Batch-Parallel Mode ({args.method}) ===")
                print(f"Loading data once and computing all distance pairs in parallel")

                # Iterate through tau thresholds
                for tau in tau_list:
                    print(f"\nProcessing tau={tau}")
                    candidates_file = f"{args.candidates_dir}/candidates_{args.dataset}_tau{int(tau)}.csv"

                    import os
                    output_dir = f"results/similarity_search/{args.dataset}/tau{tau}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_json = f"{output_dir}/{args.method}_q{args.query_start}-{args.query_end}.json"

                    trainer.similarity_search_batch_parallel(
                        candidates_file,
                        args.query_start,
                        args.query_end,
                        tau,
                        output_json=output_json
                    )
                print(f"=== Batch-Parallel Search ({args.method}) Completed ===")
            else:
                # Non-batch-parallel mode
                if args.method == "all":
                    # Three-in-one similarity search (single-threaded)
                    results = trainer.similarity_search_all_methods_optimized(
                        candidates_dir=args.candidates_dir,
                        query_start=args.query_start,
                        query_end=args.query_end,
                        tau_thresholds=tau_list,
                        output_dir="results"
                    )
                    print("=== Three-in-One Similarity Search Completed ===")
                else:
                    # Original per-query mode (GEDIOT/GEDGW/GEDHOT)
                    for tau in tau_list:
                        print(f"\nProcessing tau={tau}")
                        candidates_file = f"{args.candidates_dir}/candidates_{args.dataset}_tau{int(tau)}.csv"

                        import os
                        output_dir = f"results/similarity_search/{args.dataset}/tau{tau}"
                        os.makedirs(output_dir, exist_ok=True)
                        output_json = f"{output_dir}/{args.method}_q{args.query_start}-{args.query_end}.json"

                        trainer.similarity_search(
                            candidates_file,
                            args.query_start,
                            args.query_end,
                            tau,
                            output_json=output_json
                        )
                    print(f"=== Similarity Search ({args.method}) Completed ===")

        return  # Exit after similarity search

    # Original logic (training/testing)
    tab_printer(args)
    trainer = Trainer(args)

    if args.GW and args.model_name == "GEDGW":
        trainer.process('test')
        trainer.process('test2')
        trainer.path_score_my('test',test_k=1)
    else:
        if args.model_epoch_start > 0:
            trainer.load(args.model_epoch_start)
        if args.model_train == 1:
            for epoch in range(args.model_epoch_start, args.model_epoch_end):
                trainer.cur_epoch = epoch
                trainer.fit()
                trainer.save(epoch + 1)
                trainer.score('test')
        else:
            trainer.cur_epoch = args.model_epoch_start

            # Original testing logic
            if args.model_name == "NOAH":
                trainer.score_my('test',test_k=0)
                trainer.score_my('test2',test_k=0)
            elif not args.greedy:
                trainer.score_my('test')
                trainer.score_my('test2')
            if args.path:
                if args.model_name in ["GPN","SimGNN","TaGSim"]:
                    print("Warning: No path can be generated")
                    sys.exit()
            if args.greedy:
                trainer.path_score_my('test', test_k=1)
            elif args.model_name=="NOAH":
                trainer.path_score_my('test', test_k=0)
            else:
                #GedGNN and Ours
                trainer.path_score_my('test',test_k=100)


if __name__ == "__main__":
    main()
