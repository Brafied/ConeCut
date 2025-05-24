import pickle
import warnings
import numpy as np

warnings.filterwarnings('ignore')



from numpy.linalg import lstsq
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error


def load_all_data(features_file_name = 'features_trained_from_scratch__grammaticality'):
    print("Loading data...")
    with open(f'../data/{features_file_name}.pkl', 'rb') as f:
        features = pickle.load(f)
        features = features

    with open('../data/features_shuffleOrder_grammaticality_3053283.pkl', 'rb') as f:
        features_shuffle_order = pickle.load(f)

    with open('../data/redundant_grammaticality_3053283.pkl', 'rb') as f:
        redundant_indices = pickle.load(f)

    with open('../data/non_redundant_grammaticality_3053283.pkl', 'rb') as f:
        non_redundant_indices = pickle.load(f)

    print(f"Features shape: {features.shape}")
    # print(f"Number of redundant features: {len(redundant_indices)}")
    # print(f"Number of non-redundant features: {len(non_redundant_indices)}")

    features = np.array(features)
    redundant_indices = np.array(redundant_indices)
    non_redundant_indices = np.array(non_redundant_indices)

    return features, redundant_indices, non_redundant_indices

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
