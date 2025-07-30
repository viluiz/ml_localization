"""
Utility functions for file I/O and data handling.

@author: Vinicius Luiz Santos Silva

"""
import numpy as np 
import os
import logging

logger = logging.getLogger(__name__)


def read_binary_file(filename, nrows):
    """
    Reads a binary file and returns the data as a numpy array. Reshape the data by columns (Fortran order).
    
    Args:
        filename (str): Path to the binary file
        nrows (int): number of rows in the matrix to be read
    Returns:
        np.ndarray: Data read from the binary file
        
    Raises:
        Exception: If file reading fails
    """
    try:
        logger.info(f"Reading binary file: {filename}")
        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        logger.info(f"Binary file read! Data shape: {data.shape}\n")
        return data.reshape((nrows, -1), order='F')  
    except Exception as e:
        logger.exception(f"Error reading binary file: {filename}\n")
        raise

def save_binary_file(filename, data):
    """
    Saves a numpy array to a binary file. Write the data by columns (Fortran order).
    
    Args:
        filename (str): Path to save the binary file
        data (np.ndarray): Data to save
        
    Raises:
        Exception: If file saving fails
    """
    try:
        logger.info(f"Saving binary file: {filename}")
        with open(filename, 'wb') as f:
            np.transpose(data).astype(np.float32).tofile(f) # We transpose the data because np.tofile write the data in row order (order C). 
        logger.info(f"Binary file saved! Data shape: {data.shape}\n")

    except Exception as e:
        logger.exception(f"Error saving binary file: {filename}\n")
        raise

def validate_matrix_compatibility(X_train, Y_train, X_super):
    """
    Validates that matrices have compatible dimensions.
    
    Args:
        X_train (np.ndarray): Training input features
        Y_train (np.ndarray): Training target values
        X_super (np.ndarray): Super ensemble input features
        
    Raises:
        ValueError: If matrices have incompatible dimensions
    """
    if X_train.shape[0] != Y_train.shape[0]:
        logger.error(f"X_train and Y_train have unmatch dimension 0: {X_train.shape[0]} and {Y_train.shape[0]}\n")
        raise ValueError(f"X_train and Y_train must have same number of samples. "
                        f"Got {X_train.shape[0]} and {Y_train.shape[0]}")
    
    if X_train.shape[1] != X_super.shape[1]:
        logger.error(f"X_train and X_super have unmatch dimension 1: {X_train.shape[1]} and {X_super.shape[1]}\n")
        raise ValueError(f"X_train and X_super must have same number of features. "
                        f"Got {X_train.shape[1]} and {X_super.shape[1]}")
    
    if X_train.ndim != 2 or Y_train.ndim != 2 or X_super.ndim != 2:
        logger.error(f"X_train, Y_train, and X_super must be 2D arrays. "
                     f"Got {X_train.ndim}D, {Y_train.ndim}D, and {X_super.ndim}D arrays.\n")
        raise ValueError(f"X_train, Y_train, and X_super must be 2D arrays. "
                        f"Got {X_train.ndim}D, {Y_train.ndim}D, and {X_super.ndim}D arrays.")
    
    logger.info("Matrix compatibility validated\n")

    
def check_file_exists(filepath):
    """
    Check if a filepath exists.
    
    Args:
        filepath (str): Path to file
        
    Raises:
        Exception: If file does not exist
    """
    if not os.path.isfile(filepath):
        logger.error(f"File does not exist: {filepath}\n")
        raise FileNotFoundError(f"Error: File '{filepath}' does not exist")
    
def check_dir_exists(filepath):
    """
    Check if a directory exists.
    
    Args:
        directory (str): Path to directory
        
    Raises:
        Exception: If directory does not exist
    """
    dir = os.path.dirname(filepath)
    if not os.path.isdir(dir) and dir != '':
        logger.error(f"Directory '{dir}' does not exist: {filepath}\n")
        raise FileNotFoundError(f"Error: Directory '{dir}' does not exist for {filepath}")
    




