# import argparse
# import pickle
# import logging
# import numpy as np
# import torch
# import warnings

# # Constants
# residual_threshold = 0.001
# EPS_REL = 1e-6
# EPS_TINY = 1e-12


# warnings.filterwarnings('ignore')

# from utils import load_all_data, verify_linear_dependence

# def find_redundancy(solve_by="res", data_name = "grammar"):

#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(f'../../../logs/redundancy_{residual_threshold}_{data_name}.log'),
#             logging.StreamHandler()
#         ]
#     )
#     logger = logging.getLogger(__name__)
#     # Load data
#     features, _, _ = load_all_data(features_file_name=data_name)
#     num_examples = len(features)


#     # Initialize lists
#     non_redundant_inds = [i for i in range(num_examples)]
#     redundant_inds = []
#     results = []  # Store details for each feature

#     # Process each feature
#     for i in range(num_examples):
#         if i % 5 == 0:
#             logger.info(f"Processing feature {i + 1}/{num_examples}...")
#             logger.info(f"Current redundant count: {len(redundant_inds)}")

#         # # If no non-redundant features yet, add the first feature
#         # if len(non_redundant_inds) == 0:
#         #     non_redundant_inds =
#         #     is_zero = torch.count_nonzero(torch.from_numpy(features[i])).item() == 0
#         #     results.append({
#         #         'example_num': i,
#         #         'is_redundant': False,
#         #         'r2_score': 0,
#         #         'residual': 0,
#         #         'method': 'initial',
#         #         'is_zero': is_zero
#         #     })
#         #     continue

#         mask = np.ones(len(features), dtype=bool)
#         #do not consider every redundant element plus the current one
#         for index_to_mask in redundant_inds:
#             mask[index_to_mask] = False
#         mask[i] = False
#         X = features[mask].T

#         # print(features)
#         # print("\n\n\n")
#         # print(features[mask])
#         # print("\n\n\n")
#         # print(i)
#         # y = features[i]
#         # print(y)
#         # print("\n\n\n")
#         # print(non_redundant_inds)
#         # exit()

#         # X = features[non_redundant_inds].T
#         y = features[i]
#         # print(X.shape, y.shape)

#         # Normalize
#         X_norm = np.linalg.norm(X, axis=0, keepdims=True) + EPS_TINY
#         y_norm = np.linalg.norm(y, axis=0, keepdims=True) + EPS_TINY
#         X = X / X_norm
#         y = y / y_norm

#         # Check linear dependence
#         coeffs_nnls, _, _, _, residual_nnls = verify_linear_dependence(y, X, method='nnls')
#         coeffs_lstsq, _, _, _, residual_lstsq = verify_linear_dependence(y, X, method='lstsq')

#         # Choose method based on solve_by
#         if solve_by == "res":
#             residual = min(residual_nnls, residual_lstsq)
#             method_used = 'nnls' if residual_nnls < residual_lstsq else 'lstsq'
#         else:
#             r2_nnls = verify_linear_dependence(y, X, method='nnls')[3]
#             r2_lstsq = verify_linear_dependence(y, X, method='lstsq')[3]
#             if r2_nnls > r2_lstsq:
#                 r2 = r2_nnls
#                 method_used = 'nnls'
#                 residual = residual_nnls
#             else:
#                 r2 = r2_lstsq
#                 method_used = 'lstsq'
#                 residual = residual_lstsq
#         # Determine redundancy
#         adaptive_threshold = 1e-8 * np.linalg.norm(X)
#         is_zero = torch.count_nonzero(torch.from_numpy(y)).item() == 0



#         if solve_by == "res":
#             print(residual)
#             is_redundant = residual <= adaptive_threshold

#         else:
#             print(r2)
#             is_redundant = r2 >= 0.98

#         if is_redundant:
#             redundant_inds.append(i)

#         # Store result
#         results.append({
#             'example_num': i,
#             'is_redundant': is_redundant,
#             'residual': residual,
#             'r2_score': r2 if solve_by != "res" else None,
#             'method': method_used,
#             'is_zero': is_zero
#         })

#     # Summarize results
#     total_redundant = len(redundant_inds)
#     total_non_redundant = num_examples - total_redundant
#     print(total_non_redundant)
#     non_redundant_inds = [i for i in range(num_examples) if i not in redundant_inds]


#     logger.info("\nResults Summary:")
#     logger.info(f"Total non-redundant features: {total_non_redundant}")
#     logger.info(f"Total redundant features: {total_redundant}")
#     logger.info(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

#     residuals = [res['residual'] for res in results]

#     if solve_by != "res":
#         r2_scores = [res['r2_score'] for res in results if res['r2_score'] is not None]
#         # logger.info(f"Average R² score: {np.mean(r2_scores):.4f}")

#     zero_redundant = sum(1 for res in results if res['is_zero'] and res['is_redundant'])
#     non_zero_redundant = sum(1 for res in results if not res['is_zero'] and res['is_redundant'])
#     logger.info(f"Zero features marked redundant: {zero_redundant}")
#     logger.info(f"Non-zero features marked redundant: {non_zero_redundant}")

#     lstsq_count = sum(1 for res in results if res['method'] == 'lstsq')
#     nnls_count = sum(1 for res in results if res['method'] == 'nnls')
#     logger.info(f"Method usage: lstsq = {lstsq_count} ({lstsq_count / num_examples * 100:.2f}%), nnls = {nnls_count} ({nnls_count / num_examples * 100:.2f}%)")

#     # Save indices
#     with open(f'../../../data/non_redundant_{data_name}.pkl', 'wb') as f:
#         pickle.dump(non_redundant_inds, f)
#     with open(f'../../../data/redundant_{data_name}.pkl', 'wb') as f:
#         pickle.dump(redundant_inds, f)

#     logger.info("Non-redundant and redundant indices saved to pickle files.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Find redundant features")
#     parser.add_argument('--data_name', type=str, default="features_trained_from_scratch__grammaticality", help="Name of the dataset")
#     args = parser.parse_args()
#     find_redundancy(solve_by = "res", data_name=args.data_name)
# # find_redundancy(solve_by="r2")

import argparse
import os
import pickle
import logging
import numpy as np
import torch
import warnings

# Constants
residual_threshold = 0.001
r2_threshold = 0.95
EPS_REL = 1e-6
EPS_TINY = 1e-12


warnings.filterwarnings('ignore')

from utils import load_all_data, verify_linear_dependence

def find_redundancy(solve_by="res", data_name = "grammar", model_name = "skyworks_llama", solver = 'nnls'):
    # Use existing logger instead of creating a new one
    logger = logging.getLogger(__name__)
    logger.info("************ redundancy detection started **********************")
    logger.info(f"solving by {solve_by} for data {data_name}")
    # Load data
    try:
        features = load_all_data(features_file_name=data_name, model_name=model_name)
    except FileNotFoundError as e:
        logger.error(f"Could not find features file. Make sure features are saved first. Error: {e}")
        raise
    num_examples = len(features)

    # Initialize lists
    non_redundant_inds = [i for i in range(num_examples)]
    redundant_inds = []
    results = []  # Store details for each feature

    # Process each feature
    for i in range(num_examples):
        if i % 5 == 0:
            logger.info(f"Processing feature {i + 1}/{num_examples}...")
            logger.info(f"Current redundant count: {len(redundant_inds)}")

        # # If no non-redundant features yet, add the first feature
        # if len(non_redundant_inds) == 0:
        #     non_redundant_inds =
        #     is_zero = torch.count_nonzero(torch.from_numpy(features[i])).item() == 0
        #     results.append({
        #         'example_num': i,
        #         'is_redundant': False,
        #         'r2_score': 0,
        #         'residual': 0,
        #         'method': 'initial',
        #         'is_zero': is_zero
        #     })
        #     continue

        mask = np.ones(len(features), dtype=bool)
        #do not consider every redundant element plus the current one
        for index_to_mask in redundant_inds:
            mask[index_to_mask] = False
        mask[i] = False
        X = features[mask].T

        # print(features)
        # print("\n\n\n")
        # print(features[mask])
        # print("\n\n\n")
        # print(i)
        # y = features[i]
        # print(y)
        # print("\n\n\n")
        # print(non_redundant_inds)
        # exit()

        # X = features[non_redundant_inds].T
        y = features[i]
        # print(X.shape, y.shape)

        # Normalize
        X_norm = np.linalg.norm(X, axis=0, keepdims=True) + EPS_TINY
        y_norm = np.linalg.norm(y, axis=0, keepdims=True) + EPS_TINY
        X = X / X_norm
        y = y / y_norm

        # Choose method based on solve_by
        if solve_by == "res":
            # Check linear dependence
            coeffs_nnls, _, _, _, residual_nnls = verify_linear_dependence(y, X, method='nnls')
            coeffs_lstsq, _, _, _, residual_lstsq = verify_linear_dependence(y, X, method='lstsq')
            residual = min(residual_nnls, residual_lstsq)
            method_used = 'nnls' if residual_nnls < residual_lstsq else 'lstsq'
        else:
            if solver == "nnls":
                r2 = verify_linear_dependence(y, X, method='nnls')[3]
                method_used = 'nnls'
                residual = 'none'
            else:
                r2 = verify_linear_dependence(y, X, method='lstsq')[3]
                method_used = 'lstsq'
                residual = 'none'

        # Determine redundancy
        adaptive_threshold = 1e-8 * np.linalg.norm(X)
        is_zero = torch.count_nonzero(torch.from_numpy(y)).item() == 0
        if solve_by == "res":
            is_redundant = residual <= adaptive_threshold
        else:
            is_redundant = r2 >= r2_threshold

        if is_redundant:
            redundant_inds.append(i)

        # Store result
        results.append({
            'example_num': i,
            'is_redundant': is_redundant,
            'residual': residual,
            'r2_score': r2 if solve_by != "res" else None,
            'method': method_used,
            'is_zero': is_zero
        })

    # Summarize results
    total_redundant = len(redundant_inds)
    total_non_redundant = num_examples - total_redundant
    print(total_non_redundant)
    non_redundant_inds = [i for i in range(num_examples) if i not in redundant_inds]


    logger.info("\nResults Summary:")
    logger.info(f"Total non-redundant features: {total_non_redundant}")
    logger.info(f"Total redundant features: {total_redundant}")
    logger.info(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

    print("\nResults Summary:")
    print(f"Total non-redundant features: {total_non_redundant}")
    print(f"Total redundant features: {total_redundant}")
    print(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

    residuals = [res['residual'] for res in results]

    if solve_by != "res":
        r2_scores = [res['r2_score'] for res in results if res['r2_score'] is not None]
        # logger.info(f"Average R² score: {np.mean(r2_scores):.4f}")

    zero_redundant = sum(1 for res in results if res['is_zero'] and res['is_redundant'])
    non_zero_redundant = sum(1 for res in results if not res['is_zero'] and res['is_redundant'])
    logger.info(f"Zero features marked redundant: {zero_redundant}")
    logger.info(f"Non-zero features marked redundant: {non_zero_redundant}")

    lstsq_count = sum(1 for res in results if res['method'] == 'lstsq')
    nnls_count = sum(1 for res in results if res['method'] == 'nnls')
    logger.info(f"Method usage: lstsq = {lstsq_count} ({lstsq_count / num_examples * 100:.2f}%), nnls = {nnls_count} ({nnls_count / num_examples * 100:.2f}%)")

    # Save indices - using absolute path from current working directory
    import os
    save_dir = f'data/{model_name}_features'
    os.makedirs(save_dir, exist_ok=True)
    
    with open(f'{save_dir}/non_redundant_{data_name}_{solver}_{r2_threshold}.pkl', 'wb') as f:
        pickle.dump(non_redundant_inds, f)

    with open(f'{save_dir}/redundant_{data_name}_{solver}_{r2_threshold}.pkl', 'wb') as f:
        pickle.dump(redundant_inds, f)

    logger.info("Non-redundant and redundant indices saved to pickle files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find redundant features")
    parser.add_argument('--data_name', type=str, default="features_trained_from_scratch__grammaticality", help="Name of the dataset")
    parser.add_argument('--model_name', type=str, default="skyworks_llama", help="Name of the model")
    parser.add_argument('--solver', type=str, default="nnls", help="Name of the solver")

    args = parser.parse_args()
    os.makedirs(f"data/{args.model_name}_features", exist_ok=True)

    find_redundancy(solve_by = "r2", data_name=args.data_name, model_name=args.model_name, solver=args.solver)
