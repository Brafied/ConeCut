# import pickle
# import warnings
# import numpy as np

# warnings.filterwarnings('ignore')

# from numpy.linalg import lstsq
# from scipy.optimize import nnls
# from sklearn.metrics import mean_squared_error


# def load_all_data(features_file_name = 'features_trained_from_scratch__grammaticality'):
#     print("Loading data...")
#     with open(f'../data/{features_file_name}.pkl', 'rb') as f:
#         features = pickle.load(f)
#         features = features

#     with open('../data/features_shuffleOrder_grammaticality_3053283.pkl', 'rb') as f:
#         features_shuffle_order = pickle.load(f)

#     with open('../data/redundant_grammaticality_3053283.pkl', 'rb') as f:
#         redundant_indices = pickle.load(f)

#     with open('../data/non_redundant_grammaticality_3053283.pkl', 'rb') as f:
#         non_redundant_indices = pickle.load(f)

#     print(f"Features shape: {features.shape}")
#     # print(f"Number of redundant features: {len(redundant_indices)}")
#     # print(f"Number of non-redundant features: {len(non_redundant_indices)}")

#     features = np.array(features)
#     redundant_indices = np.array(redundant_indices)
#     non_redundant_indices = np.array(non_redundant_indices)

#     return features, redundant_indices, non_redundant_indices

# def verify_linear_dependence(feature_vector, basis_vectors, method='nnls'):

#     X = basis_vectors
#     y = feature_vector

#     if method == 'nnls':
#         coefficients, residuals = nnls(X, y)
#         reconstruction = X @ coefficients

#     elif method == 'lstsq':
#         coefficients, residuals, rank, s = lstsq(X, y, rcond=None)
#         reconstruction = X @ coefficients

#     mse = mean_squared_error(y, reconstruction)
#     ss_total = np.sum((y - np.mean(y))**2)
#     ss_residual = np.sum((y - reconstruction)**2)


#     r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
#     if ss_total == 0 and np.allclose(y, 0, atol=1e-10):
#         r2_score = 1
#     return coefficients, reconstruction, mse, r2_score, residuals
import logging
import pickle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from numpy.linalg import lstsq
from scipy.optimize import nnls
import warnings
warnings.filterwarnings('ignore')

def load_all_data(features_file_name = 'features_trained_from_scratch__grammaticality', model_name = 'skyworks_llama'):
    print("Loading data...")
    print(f"loading {features_file_name}")
    logging.info(f" loading features from the file : data/{model_name}_features/{features_file_name}.pkl")
    with open(f'data/{model_name}_features/{features_file_name}.pkl', 'rb') as f:
        features = pickle.load(f)
    logging.info(f"shape is {features.shape}")

    # # Optional files - only load if they exist
    # try:
    #     with open('../data/features_shuffleOrder_grammaticality_3053283.pkl', 'rb') as f:
    #         features_shuffle_order = pickle.load(f)
    # except FileNotFoundError:
    #     features_shuffle_order = None
    #     logging.info("features_shuffle_order file not found, skipping")

    # try:
    #     with open('../data/redundant_grammaticality_3053283.pkl', 'rb') as f:
    #         redundant_indices = pickle.load(f)
    # except FileNotFoundError:
    #     redundant_indices = np.array([])
    #     logging.info("redundant_indices file not found, using empty array")

    # try:
    #     with open('../data/non_redundant_grammaticality_3053283.pkl', 'rb') as f:
    #         non_redundant_indices = pickle.load(f)
    # except FileNotFoundError:
    #     non_redundant_indices = np.array([])
    #     logging.info("non_redundant_indices file not found, using empty array")

    # print(f"Features shape: {features.shape}")
    # print(f"Number of redundant features: {len(redundant_indices)}")
    # print(f"Number of non-redundant features: {len(non_redundant_indices)}")

    # Convert features to numpy array
    # try:
    #     if hasattr(features, 'numpy'): 
    #         features = features.cpu().numpy()
    #     elif not isinstance(features, np.ndarray):
    #         features = np.array(features)
    # except Exception as e:
    #     print(f"Error converting features to numpy: {e}")

    # try:
    #     features = features.astype(np.float32)
    # except Exception as e:
    #     print(f"Error converting features to float32: {e}")

    try:
        features = np.array(features)
    except Exception as e :
        print(e)

    try:
        features = features.to(torch.float32).cpu().numpy()
    except Exception as e:
        print(e)

    # redundant_indices = np.array(redundant_indices)
    # non_redundant_indices = np.array(non_redundant_indices)

    return features

def verify_linear_dependence(feature_vector, basis_vectors, method='nnls'):

    X = basis_vectors
    y = feature_vector

    if method == 'nnls':
        coefficients, residuals = nnls(X, y)
        reconstruction = X @ coefficients

    elif method == 'lstsq':
        coefficients, residuals, rank, s = lstsq(X, y, rcond=None)
        reconstruction = X @ coefficients

    mse = mean_squared_error(y, reconstruction)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - reconstruction)**2)


    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    if ss_total == 0 and np.allclose(y, 0, atol=1e-10):
        r2_score = 1
    return coefficients, reconstruction, mse, r2_score, residuals
