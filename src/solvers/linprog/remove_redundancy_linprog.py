import pickle
import logging
import numpy as np
from scipy.optimize import linprog
import argparse

EPSILON = 0.01
UPPER_BOUND = 100000

from utils import load_all_data

def find_redundancy(data_name = "grammar"):
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'../../../logs/redundancy_linprog_{data_name}_.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    features, _, _ = load_all_data(features_file_name=data_name)
    num_examples = len(features)

    non_redundant_inds = []
    results = []
    redundant_inds = []

    for i in range(num_examples):
        if i % 25 == 0:
            logger.info(f"Processing feature {i + 1}/{num_examples}...")
            logger.info(f"Current redundant count: {len(redundant_inds)}")

        mask = np.ones(len(features), dtype=bool)
        # do not consider every redundant element plus the current one
        for index_to_mask in redundant_inds:
            mask[index_to_mask] = False
        mask[i] = False

        feats_to_consider = features[mask]
        feature_to_test = features[i]

        # LP setup
        # Objective: minimize w.dot(feature_to_test)
        c = feature_to_test


        ##### this part is not necessarily correct
        A_ub_lower = -feats_to_consider
        b_ub_lower = np.ones(len(feats_to_consider)) * EPSILON  # Note: -(-EPSILON) = EPSILON
        feature_dim = features.shape[1]

        A_ub_upper = feats_to_consider
        b_ub_upper = np.ones(len(feats_to_consider)) * UPPER_BOUND

        A_ub = np.vstack([A_ub_lower, A_ub_upper])
        b_ub = np.concatenate([b_ub_lower, b_ub_upper])
        # Add constraint to prevent trivial solution: sum(w) = 1
        A_eq = np.ones((1, feature_dim))
        b_eq = np.array([1.0])

        bounds = None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs')
        ##### this part is not necessarily correct
        if res.success:
            min_value = res.fun
            optimal_w = res.x
            print(min_value)
            is_redundant = round(min_value, 6) >= -EPSILON
        else:
            is_redundant = False
            min_value = None

        if is_redundant:
            redundant_inds.append(i)

        # Store result
        results.append({
            'example_num': i,
            'is_redundant': is_redundant,
            'min_value': min_value,
            'method': 'linprog'
        })

    total_redundant = len(redundant_inds)
    total_non_redundant = num_examples - total_redundant
    non_redundant_inds = [i for i in range(num_examples) if i not in redundant_inds]


    logger.info("\nResults Summary:")
    logger.info(f"Total non-redundant features: {total_non_redundant}")
    logger.info(f"Total redundant features: {total_redundant}")
    logger.info(f"Percentage redundant: {(total_redundant / num_examples) * 100:.2f}%")

    # Save indices
    with open(f'../../../data/non_redundant_linprog_{data_name}.pkl', 'wb') as f:
        pickle.dump(non_redundant_inds, f)
    with open(f'../../../data/redundant_linprog_{data_name}.pkl', 'wb') as f:
        pickle.dump(redundant_inds, f)

    logger.info("Non-redundant and redundant indices saved to pickle files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find redundant features")
    parser.add_argument('--data_name', type=str, default="grammar", help="Name of the dataset")
    args = parser.parse_args()
    find_redundancy(data_name=args.data_name)
    # find_redundancy()