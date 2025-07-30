"""
Localization functions to generate the localization matrix.

@author: Vinicius Luiz Santos Silva

"""
import numpy as np 
import logging
import time

logger = logging.getLogger(__name__)


def crosscov(M,D):
    """
    Calculates the cross-covariance matrix between two datasets.

    Args:
        M (np.ndarray): First dataset (shape: n_parameters x n_samples)
        D (np.ndarray): Second dataset (shape: n_obsdata x n_samples)
    Returns:
        np.ndarray: Cross-covariance matrix (shape: n_parameters x n_obsdata)
    Raises:
        ValueError: If the number of samples in X and Y do not match and if they are not 2D arrays
    """
    logger.info("Calculating cross-covariance...")
    start_time = time.time()

    if M.shape[1] != D.shape[1]:
        logger.error(f"M and D have unmatch number of samples: {M.shape[1]} and {D.shape[1]}\n")
        raise ValueError(f"M and D must have the same number of samples. Got {M.shape[1]} and {D.shape[1]}")
    if M.ndim != 2 or D.ndim != 2:
        logger.error(f"M and D must be 2D arrays. Got {M.ndim}D and {D.ndim}D arrays.\n")
        raise ValueError(f"M and D must be 2D arrays. Got {M.ndim}D and {D.ndim}D arrays.")

    Cmd = (M.T-M.mean(axis=1)).T@(D.T-D.mean(axis=1))/(D.shape[1]-1)

    end_time = time.time()
    logger.info(f"Cross-covariance calculed! Calculation time = {end_time - start_time:.4f} s\n")

    return Cmd

def po_localization(Cmd, m_var, d_var, Ne, epsilon=1e-3):
    """
    Calculates the localization matrix using the Furrer and Bengtsoon localization method.

    Args:
        Cmd (np.ndarray): Cross-covariance matrix (shape: n_parameters x n_obsdata)
        m_var (np.ndarray): Variance of the first dataset (shape: n_parameters x 1)
        d_var (np.ndarray): Variance of the second dataset (shape: n_obsdata x 1)
        Ne (int): Original ensemble size
        epsilon (float, optional): For correlations below epsilon localization is set to zero. Defaults to 1e-3.
    Returns:
        np.ndarray: Localization matrix (shape: n_parameters x n_obsdata)
    Raises:
        ValueError: If the dimensions of Cmd, m_var, and d_var do not match
    """
    logger.info("Calculating PO Localization...")
    start_time = time.time()

    if Cmd.shape[0] != m_var.shape[0] or Cmd.shape[1] != d_var.shape[0]:
        logger.error(f"Cmd, m_var, and d_var have unmatch dimensions: {Cmd.shape}, {m_var.shape}, {d_var.shape}\n")
        raise ValueError(f"Cmd, m_var, and d_var must have compatible dimensions. Got Cmd: {Cmd.shape}, m_var: {m_var.shape}, d_var: {d_var.shape}")

    Cmd2 = Cmd**2
    Cmmdd = m_var@d_var.T
    R = Cmd2/(Cmd2 + (Cmd2+Cmmdd)/Ne)
    R[abs(Cmd) < epsilon*np.sqrt(Cmmdd)] = 0.0

    if np.isnan(R).any():
        logger.warning("Localization matrix contains NaN values. Replacing with 0.")
    if np.isinf(R).any():
        logger.warning("Localization matrix contains infinite values. Replacing with 1.")
    R = np.nan_to_num(R, nan=0.0, posinf=1.0, neginf=1.0)

    end_time = time.time()
    logger.info(f"PO Localization calculated! Calculation time = {end_time - start_time:.4f} s\n")

    return R


def calculate_localization(M, D, Ne, epsilon=1e-3):
    """
    Generates the localization matrix.

    Args:
        M (np.ndarray): First dataset (shape: n_parameters x n_samples)
        D (np.ndarray): Second dataset (shape: n_obsdata x n_samples)
        Ne (int): Original ensemble size
        epsilon (float, optional): For correlations below epsilon localization is set to zero. Defaults to 1e-3.
    Returns:
        np.ndarray: Localization matrix (shape: n_parameters x n_obsdata)
    """
    Cmd = crosscov(M, D)
    m_var = M.var(axis=1, ddof=1).reshape(-1, 1)
    d_var = D.var(axis=1, ddof=1).reshape(-1, 1)

    return po_localization(Cmd, m_var, d_var, Ne, epsilon)


