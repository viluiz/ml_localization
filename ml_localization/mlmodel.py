"""
Machine learning functions for training and prediction

@author: Vinicius Luiz Santos Silva

"""
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import logging

logger = logging.getLogger(__name__)


def scale_dataset(X_train, Y_train): # Row matrices
    """ 
    Scales the training dataset using MinMaxScaler to the range (-1, 1).

    Args:
        X_train (np.ndarray): Training input data (shape: n_samples x n_features)
        Y_train (np.ndarray): Training output data (shape: n_samples x n_outputs)
    Returns:
        X_train_scaled (np.ndarray): Scaled training input data (shape: n_samples x n_features)
        Y_train_scaled (np.ndarray): Scaled training output data (shape: n_samples x n_outputs)
        xnorm_scaler (Pipeline): Scaler used for input data normalization
        ynorm_scaler (Pipeline): Scaler used for output data normalization
    """
    logger.info("Scaling dataset...")
    start = time.time()

    xnorm_scaler = Pipeline([('minmax_scaler', MinMaxScaler((-1,1))),])
    ynorm_scaler = Pipeline([('minmax_scaler', MinMaxScaler((-1,1))),])
    
    X_train_scaled = xnorm_scaler.fit_transform(X_train)
    Y_train_scaled = ynorm_scaler.fit_transform(Y_train)

    end_training = time.time()
    logger.info(f"Dataset scaled! Scaling time = {end_training - start:.4f} s\n")

    return X_train_scaled, Y_train_scaled, xnorm_scaler, ynorm_scaler


def train_ml_model(X_train_scaled, Y_train_scaled): 
    """ 
    Trains the ML model using the scaled training data.

    Args:
        X_train_scaled (np.ndarray): Scaled training input data (shape: n_samples x n_features)
        Y_train_scaled (np.ndarray): Scaled training output data (shape: n_samples x n_outputs)
    Returns:
        ml_model (MultiOutputRegressor): Trained ML model
    """
    logger.info("Training ML model...")
    start = time.time()

    ml_model = MultiOutputRegressor(lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1), n_jobs=-1 )
    # logger.info(f"ML model parameters: {ml_model.get_params()}\n")
    ml_model.fit(X_train_scaled, Y_train_scaled)

    end_training = time.time() 
    logger.info(f"ML trained! Training time = {end_training - start:.4f} s")
    logger.info(f"ML training rmse [outputs range (-1, 1)]= {root_mean_squared_error(Y_train_scaled, ml_model.predict(X_train_scaled)):.4f}\n")

    return ml_model

def generate_super_ensemble(X_super, ml_model, xnorm_scaler, ynorm_scaler):
    """ 
    Generates the super ensemble predictions using the trained ML model.

    Args:
        X_super (np.ndarray): Input data for super ensemble prediction (shape: n_samples x n_features)
        ml_model (MultiOutputRegressor): Trained ML model
        xnorm_scaler (Pipeline): Scaler used for input data normalization
        ynorm_scaler (Pipeline): Scaler used for output data normalization
    Returns:
        D_super (np.ndarray): Super ensemble predictions (shape: n_samples x n_outputs)
    """
    logger.info("Generating super ensemble predictions...")
    start_time = time.time()

    Y_super = ynorm_scaler.inverse_transform(ml_model.predict(xnorm_scaler.transform(X_super)))

    end_time = time.time()
    logger.info(f"Super ensemble predictions generated! Prediction time = {end_time - start_time:.4f} s\n")

    return Y_super

def run_mllocalization_pipeline(X_train, Y_train, X_super):
    """ 
    Build pipiline for scaling, training and prediction.
    Args:
        X_train (np.ndarray): Training input data (shape: n_samples x n_features)
        Y_train (np.ndarray): Training output data (shape: n_samples x n_outputs)
        X_super (np.ndarray): Input data for super ensemble prediction (shape: n_samples x n_features)
    Returns:
        D_super (np.ndarray): Super ensemble predictions (shape: n_samples x n_outputs)
    """
    X_train_scaled, Y_train_scaled, xnorm_scaler, ynorm_scaler = scale_dataset(X_train, Y_train)
    ml_model = train_ml_model(X_train_scaled, Y_train_scaled)
    Y_super = generate_super_ensemble(X_super, ml_model, xnorm_scaler, ynorm_scaler)
    
    return Y_super





