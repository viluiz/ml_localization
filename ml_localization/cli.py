"""
Command Line Interface code and main() function

@author: Vinicius Luiz Santos Silva

"""
import argparse
import logging
import os

from .utils import read_binary_file, save_binary_file, check_file_exists, check_dir_exists, validate_matrix_compatibility
from .mlmodel import run_mllocalization_pipeline
from .localization import calculate_localization

logger = logging.getLogger(__name__)

def main():
     # Parse command line arguments
    parser = argparse.ArgumentParser(description="To run from module 'python -m ml_localization <args>' or from anywhere:  'python <module path>/run.py <args>'")
    parser.add_argument("-m","--nm", type=int, required=True, help="Number of model parameters.")
    parser.add_argument("-d","--nd", type=int, required=True, help="Number of observed data points.")
    parser.add_argument("-M","--mfile", type=str, required=True, help="Filepath binary file (float32 - Fortran order) of matrix M of size: n_parameters x ensemble_size.")
    parser.add_argument("-D","--dfile", type=str, required=True, help="Filepath binary file (float32 - Fortran order) of matrix D of size: n_obsdata x ensemble_size.")
    parser.add_argument("-Ms","--msfile", type=str, required=True, help="Filepath binary file (float32 - Fortran order) of super matrix M of size: n_parameters x super_ensemble_size.")
    parser.add_argument("-Ds","--dsfile", type=str, default='', help="Filepath to save binary file (float32 - Fortran order) of super matrix D of size: n_obsdata x super_ensemble_size. If not given it will not be saved.")
    parser.add_argument("-R","--rfile", type=str, default='', help="Filepath to save binary file (float32 - Fortran order) of localization matrix of size: n_parameters x n_obsdata. If not given it will not be saved/calculated.")
    parser.add_argument("-l","--logfile", type=str, default="ml_localization.log", help="Log file path.")
    
    args = parser.parse_args()

    # Configure logging
    check_dir_exists(args.logfile)
    logging.basicConfig(filename=args.logfile, 
                        level=logging.INFO , 
                        format="{asctime} - {name}:{levelname} - {message}",
                        style="{",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.info("---------------------------------------")
    logger.info("Developed by Vinicius Luiz Santos Silva (BFU0)")
    logger.info(f"Running from: {os.getcwd()}")
    logger.info("Starting ML localization process...\n")
    

    # IO: checking
    check_file_exists(args.mfile)
    check_file_exists(args.dfile)
    check_file_exists(args.msfile)
    
    # IO: reading
    X_train = read_binary_file(args.mfile, args.nm).T # X_train = M.T
    Y_train = read_binary_file(args.dfile, args.nd).T # Y_train = D.T
    X_super = read_binary_file(args.msfile, args.nm).T # X_super = M_super.T

    # Validate data 
    logger.info(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}, X_super shape: {X_super.shape}\n")
    validate_matrix_compatibility(X_train, Y_train, X_super) 

    # run machine learning pipeline
    Y_super = run_mllocalization_pipeline(X_train, Y_train, X_super)

    # Saving and localization matrix calculation
    if args.dsfile != "":
        check_dir_exists(args.dsfile)
        save_binary_file(args.dsfile, Y_super.T) # D_super = Y_super.T
    if args.rfile != "":
        check_dir_exists(args.rfile)
        # Calculate localization matrix
        R = calculate_localization(M=X_super.T, D=Y_super.T, Ne=X_train.shape[0]) 
        save_binary_file(args.rfile, R)

    logger.info("ML localization process completed successfully!!!\n")


if __name__ == "__main__":
    main()



