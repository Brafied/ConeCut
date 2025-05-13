
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from numpy.linalg import lstsq, matrix_rank
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
with open('./data/features_grammaticality.pkl', 'rb') as f:
    features = pickle.load(f)

with open('./data/features_shuffleOrder_grammaticality_3053283.pkl', 'rb') as f:
    features_shuffle_order = pickle.load(f)

features = features[features_shuffle_order]

with open('./data/redundant_grammaticality_3053283.pkl', 'rb') as f:
    redundant_indices = pickle.load(f)
    
with open('./data/non_redundant_grammaticality_3053283.pkl', 'rb') as f:
    non_redundant_indices = pickle.load(f)

print(f"Features shape: {features.shape}")
print(f"Number of redundant features: {len(redundant_indices)}")
print(f"Number of non-redundant features: {len(non_redundant_indices)}")

features = np.array(features)
redundant_indices = np.array(redundant_indices)
non_redundant_indices = np.array(non_redundant_indices)

def verify_linear_dependence(feature_vector, basis_vectors, method='ols'):
    """
    Check if feature_vector can be expressed as a linear combination of basis_vectors
    
    Parameters:
    - feature_vector: The vector to be represented
    - basis_vectors: The set of vectors to use for representation
    - method: 'nnls', or 'lstsq'
    
    Returns:
    - coefficients: The coefficients of the linear combination
    - reconstruction: The reconstructed vector
    - error: The mean squared error between the original and reconstruction
    - r2_score: The R² score (coefficient of determination)
    """
    X = basis_vectors
    y = feature_vector
    
    if method == 'nnls':
        # Non-negative least squares
        coefficients, residuals = nnls(X, y)
        reconstruction = X @ coefficients
        
    elif method == 'lstsq':
        # Direct linear least squares using numpy
        coefficients, residuals, rank, s = lstsq(X, y, rcond=None)
        reconstruction = X @ coefficients
    
    mse = mean_squared_error(y, reconstruction)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum((y - reconstruction)**2)
    r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    if ss_total == 0 and np.allclose(y, 0, atol=1e-10):
        r2_score = 1
        # all zero feature
    return coefficients, reconstruction, mse, r2_score

redundant_features = features[redundant_indices]
non_redundant_features = features[non_redundant_indices]

print("\nAnalyzing redundancy...")

# Prepare data for regression
# We'll transpose to have each feature as a row and each dimension as a column
X = non_redundant_features.T  # Each column is a non-redundant feature
num_redundant = len(redundant_indices)

# Results storage
results = []
r2_scores = []
mse_scores = []
sparsity_levels = []  # To track how many non-redundant features are used

# Analyze each redundant feature
for i, redundant_idx in enumerate(redundant_indices):
    if i % 50 == 0:
        print(f"Processing redundant feature {i+1}/{num_redundant}...")
    
    y = features[redundant_idx]  # The redundant feature we're trying to reconstruct

    # Try both methods for comparison
    coeffs_nnls, reconstruction_nnls, mse_nnls, r2_nnls = verify_linear_dependence(y, X, method='nnls')
    coeffs_lstsq, reconstruction_lstsq, mse_lstsq, r2_lstsq = verify_linear_dependence(y, X, method='lstsq')
    
    # Use whichever method performed better
    if r2_lstsq > r2_nnls:
        coeffs = coeffs_lstsq
        reconstruction = reconstruction_lstsq
        mse = mse_lstsq
        r2 = r2_lstsq
        method_used = 'lstsq'
    else:
        coeffs = coeffs_nnls
        reconstruction = reconstruction_nnls
        mse = mse_nnls
        r2 = r2_nnls
        method_used = 'nnls'

    
    # Calculate how many non-redundant features are used (non-zero coefficients)
    non_zero_coeffs = np.sum(np.abs(coeffs) > 1e-5)
    sparsity = non_zero_coeffs / len(non_redundant_indices)
    
    results.append({
        'redundant_idx': redundant_idx,
        'r2_score': r2,
        'mse': mse,
        'non_zero_coeffs': non_zero_coeffs,
        'coeffs': coeffs,
        'method': method_used
    })
    
    r2_scores.append(r2)
    mse_scores.append(mse)
    sparsity_levels.append(sparsity)

# Analyze results
r2_threshold = 1  # perfect match
print(r2_scores)
high_r2_count = sum(r2 >= r2_threshold for r2 in r2_scores)
print(high_r2_count)
percent_explained = (high_r2_count / num_redundant) * 100
print(percent_explained)

# check for both nnls and lstsq
lstsq_count = sum(1 for result in results if result['method'] == 'lstsq')
nnls_count = sum(1 for result in results if result['method'] == 'nnls')

print("\nResults Summary:")
print(f"Total redundant features analyzed: {num_redundant}")
print(f"Features with R² ≥ {r2_threshold}: {high_r2_count} ({percent_explained:.2f}%)")
print(f"Average R² score: {np.mean(r2_scores):.4f}")
print(f"Average MSE: {np.mean(mse_scores):.6f}")
print(f"Average percentage of non-redundant features used: {np.mean(sparsity_levels)*100:.2f}%")
print(f"Method usage: lstsq = {lstsq_count} ({lstsq_count/num_redundant*100:.2f}%), nnls = {nnls_count} ({nnls_count/num_redundant*100:.2f}%)")

r2_array = np.array(r2_scores)
best_indices = np.argsort(r2_array)[-5:][::-1]
worst_indices = np.argsort(r2_array)[:5]

best_idx = best_indices[0]

best_coeffs = results[best_idx]['coeffs']
significant_coeffs = [(non_redundant_indices[i], coeff) for i, coeff in enumerate(best_coeffs) if coeff > 1e-5]

print(f"Top contributing non-redundant features:")
for i, (feat_idx, coeff) in enumerate(significant_coeffs[:10]):  # Show top 10 contributors
    print(f"  - Feature {feat_idx}: coefficient = {coeff:.4f}")

print("\nConclusion:")
if percent_explained > 90:
    print(f"STRONG EVIDENCE OF REDUNDANCY: {percent_explained:.2f}% of the supposedly redundant features can be accurately reconstructed (R² ≥ {r2_threshold}) as linear combinations of non-redundant features.")
else:
    print(f"WEAK EVIDENCE OF REDUNDANCY: Only {percent_explained:.2f}% of the supposedly redundant features can be accurately reconstructed (R² ≥ {r2_threshold}) as linear combinations of non-redundant features.")