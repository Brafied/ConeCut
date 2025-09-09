import os, argparse, json, pickle as pkl, logging, glob
from collections import Counter
import numpy as np
import torch
import warnings
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from sklearn.metrics import mean_squared_error
from numpy.linalg import lstsq
from scipy.optimize import nnls

warnings.filterwarnings('ignore')

os.environ["HF_HOME"] = "/scratch/general/vast/u1472659/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u1472659/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"]  = "/scratch/general/vast/u1472659/huggingface_cache/datasets"
CACHE_DIR = "/scratch/general/vast/u1472659/huggingface_cache/"

from reward_bench_adapter import run_rm_subset    
from reward_model_inference_utils import (
    run_redundancy_tests, calculate_accuracy
)
from solvers.nnls.remove_redundancy_nnls import find_redundancy

SECTION_MAP = {     
    "chat":       "Chat",
    "chat_hard":  "Chat Hard",
    "safety":     "Safety",
    "reasoning":  "Reasoning",
}

# Gold standard model for redundancy detection
GOLD_STANDARD_MODEL = "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1"

def check_features_exist(model_name, subset_suffix):
    """
    Check if features already exist for the given model and subset.
    
    Args:
        model_name: Short name of the model (e.g., 'QRM-Gemma-2-27B')
        subset_suffix: The subset suffix (e.g., '_safety', '_reasoning', '_full')
        
    Returns:
        tuple: (features_exist, features_path) where features_exist is boolean and
               features_path is the path to the existing features file
    """
    data_dir = f"data/{model_name}_features"
    features_diff_file = f"features_diff{subset_suffix}.pkl"
    features_path = os.path.join(data_dir, features_diff_file)
    
    return os.path.exists(features_path), features_path

def load_existing_features(model_name, subset_suffix):
    """
    Load existing features and related data if they exist.
    
    Args:
        model_name: Short name of the model
        subset_suffix: The subset suffix
        
    Returns:
        tuple: (feat_diff, scores_ch, scores_rj, subsets) or (None, None, None, None) if not found
    """
    # Check main features directory first
    features_exist, features_path = check_features_exist(model_name, subset_suffix)
    
    if not features_exist:
        return None, None, None, None
    
    # Use consistent directory structure - look in data directory for both features and scores
    data_dir = f"data/{model_name}_features"
    
    scores_ch_path = os.path.join(data_dir, f"scores_chosen{subset_suffix}.pkl")
    scores_rj_path = os.path.join(data_dir, f"scores_rejected{subset_suffix}.pkl")
    
    if not (os.path.exists(scores_ch_path) and os.path.exists(scores_rj_path)):
        print(f"Score files not found in {data_dir}, will regenerate features")
        return None, None, None, None
    
    try:
        # Load features
        with open(features_path, "rb") as f:
            feat_diff = pkl.load(f)
        
        # Load scores
        with open(scores_ch_path, "rb") as f:
            scores_ch = pkl.load(f)
        
        with open(scores_rj_path, "rb") as f:
            scores_rj = pkl.load(f)
        
        print(f"Found existing features for {model_name} with subset {subset_suffix}")
        print(f"Features shape: {feat_diff.shape}, Scores length: {len(scores_ch)}")
        
        # Check for dimension mismatch between features and scores
        if feat_diff.shape[0] != len(scores_ch) or feat_diff.shape[0] != len(scores_rj):
            print(f" Dimension mismatch detected:")
            print(f"   Features ({subset_suffix}): {feat_diff.shape[0]} examples")
            print(f"   Chosen scores: {len(scores_ch)} examples") 
            print(f"   Rejected scores: {len(scores_rj)} examples")
            print(f"   ")
            print(f"  This happens because:")
            print(f"  Features are subset-specific: data/{model_name}_features/features_diff{subset_suffix}.pkl")
            print(f"  Scores are cached: {data_dir}/scores_*.pkl")
            print(f"  Generic scores are from the last run (different subset)")
            print(f"   ")
            print(f"  Will regenerate features to ensure consistency")
            logging.warning(f"Dimension mismatch - Features: {feat_diff.shape[0]}, Scores: {len(scores_ch)}, {len(scores_rj)}")
            return None, None, None, None
        
        return feat_diff, scores_ch, scores_rj, None  # subsets will be None, we'll get them from evaluation
        
    except Exception as e:
        print(f"Error loading existing features: {e}")
        return None, None, None, None

def get_gold_standard_redundant_indices(subset_suffix, solver, threshold):
    """
    Get redundant indices from the gold standard model if they exist.
    
    Args:
        subset_suffix: The subset suffix (e.g., '_safety', '_reasoning', '_full')
        solver: The solver used (e.g., 'nnls', 'lstsq')
        threshold: The threshold used (e.g., 0.95)
        
    Returns:
        tuple: (redundant_indices, path_exists) where redundant_indices is the list of indices
               or None if not found, and path_exists is boolean indicating if file exists
    """
    gold_standard_name = GOLD_STANDARD_MODEL.split('/')[-1]
    gold_standard_path = f"data/{gold_standard_name}_features/redundant_features_diff{subset_suffix}_{solver}_{threshold}.pkl"
    
    if os.path.exists(gold_standard_path):
        try:
            with open(gold_standard_path, "rb") as f:
                redundant_indices = pkl.load(f)
            return redundant_indices, True
        except Exception as e:
            logging.warning(f"Failed to load gold standard redundant indices from {gold_standard_path}: {e}")
            return None, False
    else:
        return None, False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1", 
                    help="Model ID to evaluate (default: ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1)")
    ap.add_argument("--filter_by_subset", default="", choices=list(SECTION_MAP.keys()) + [""],
                    help="Top-level RewardBench section to keep")
    ap.add_argument("--eval_only",   action="store_true")
    ap.add_argument("--find_redundancy", action="store_true", help="Find redundant features after feature extraction")
    ap.add_argument("--redundancy_solver", default="nnls", choices=["nnls", "lstsq"], help="Solver for redundancy detection")
    ap.add_argument("--redundancy_threshold", default=0.95, type=float, help="R2 threshold for redundancy detection")
    ap.add_argument("--force_regenerate", action="store_true", help="Force regeneration of features even if cache exists")
    ap.add_argument("--skip_cache", action="store_true", help="Skip cache entirely and always run fresh inference")
    return ap.parse_args()


def main():
    args = parse_args()
    MODEL_ID = args.model_id
    model_name_short = MODEL_ID.split('/')[-1]
    os.makedirs(f"data/{model_name_short}_features", exist_ok=True)

    LOGDIR   = os.path.expanduser("~/alignment_benchmark_LLM/logging/")
    os.makedirs(LOGDIR, exist_ok=True)
    
    log_suffix = f"_{args.redundancy_threshold}" if args.find_redundancy else ""
    log_filename = f"{MODEL_ID.split('/')[-1]}_{args.filter_by_subset or 'all'}{log_suffix}.txt"
    
    logging.basicConfig(
        filename=os.path.join(LOGDIR, log_filename),
        level=logging.INFO,
    )

    subset_suffix = f"_{args.filter_by_subset}" if args.filter_by_subset else "_full"
    
    if args.force_regenerate or args.skip_cache:
        action = "Force regeneration" if args.force_regenerate else "Skip cache"
        print(f"🔄 {action} requested - skipping cache check")
        logging.info(f"{action} requested - skipping cache check")
        feat_diff, scores_ch, scores_rj, cached_subsets = None, None, None, None
    else:
        feat_diff, scores_ch, scores_rj, cached_subsets = load_existing_features(model_name_short, subset_suffix)
    
    if feat_diff is not None and scores_ch is not None and scores_rj is not None:
        print(f" Using cached features for {MODEL_ID}")
        print(f"   Features shape: {feat_diff.shape}")
        print(f"   Skipping model loading and inference")
        logging.info(f"Using cached features for {MODEL_ID} - skipping model loading and inference")
        
        subsets = None
            
        h_ch = None  
        h_rj = None  
    else:
        print(f"🔄 No cached features found for {MODEL_ID}")
        print(f"   Running full model inference...")
        logging.info(f"No cached features found for {MODEL_ID} - running full model inference")
        
        scores_ch, scores_rj, h_ch, h_rj, subsets = run_rm_subset(
            MODEL_ID,
            section=SECTION_MAP.get(args.filter_by_subset, None),
            batch_size=8,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR
        )

    acc, n_correct, scores_ch_arr, scores_rj_arr, correct_idx = calculate_accuracy(scores_ch, scores_rj)
    logging.info(f"unweighted Accuracy: {acc:.4f} ({n_correct}/{len(scores_ch)})")

    if subsets is None:
        print("Cached features detected - need to get proper subset information")
        print("Running minimal dataset loading to get correct subset labels...")
        logging.info("Loading dataset to get proper subset information for cached features")
        
        try:
            from rewardbench import load_eval_dataset
            dataset, subsets_only = load_eval_dataset(
                core_set=True,
                conv=None,  # We don't need conversation templates
                custom_dialogue_formatting=False,
                tokenizer=None,  # We don't need tokenizer
                logger=None,
                keep_columns=["subset"],  # Only keep subset information
                max_turns=None
            )
            subsets_only = [item["subset"] for item in dataset]
            
            # Filter by subset if specified
            if args.filter_by_subset and args.filter_by_subset in SECTION_MAP:
                section_name = SECTION_MAP[args.filter_by_subset]
                expected_subsets = SUBSET_MAPPING[section_name]
                # Filter to only include the expected subsets
                filtered_indices = [i for i, s in enumerate(subsets_only) if s in expected_subsets]
                
                if len(filtered_indices) == len(scores_ch):
                    subsets = [subsets_only[i] for i in filtered_indices]
                    print(f"Successfully reconstructed {len(subsets)} proper subset labels")
                else:
                    print(f"Subset filtering mismatch:")
                    print(f"Original dataset: {len(subsets_only)} examples")
                    print(f"Filtered subset: {len(filtered_indices)} examples") 
                    print(f"Cached scores: {len(scores_ch)} examples")
                    print(f"This suggests cached data is already filtered - reconstructing proper subset labels")
                    # The cached data is already subset-filtered, so we need to reconstruct the actual subset labels
                    # Get the expected subsets for this section
                    section_name = SECTION_MAP[args.filter_by_subset]
                    expected_subsets = SUBSET_MAPPING[section_name]
                    
                    # Load the full dataset to get the correct subset distribution
                    from rewardbench import load_eval_dataset
                    full_dataset, full_subsets = load_eval_dataset(
                        core_set=True,
                        conv=None,
                        custom_dialogue_formatting=False,
                        tokenizer=None,
                        logger=None,
                        keep_columns=["subset"],
                        max_turns=None
                    )
                    full_subsets = [item["subset"] for item in full_dataset]
                    
                    # Filter to get the subset distribution for this section
                    section_indices = [i for i, s in enumerate(full_subsets) if s in expected_subsets]
                    section_subsets = [full_subsets[i] for i in section_indices]
                    
                    if len(section_subsets) == len(scores_ch):
                        subsets = section_subsets
                        print(f"Successfully reconstructed proper subset labels: {set(subsets)}")
                    else:
                        print(f"Still mismatch - using proportional reconstruction")
                        # Use proportional reconstruction based on expected subset counts
                        subset_counts = Counter([s for s in full_subsets if s in expected_subsets])
                        total_section_examples = sum(subset_counts.values())
                        
                        # Calculate proportions and distribute examples
                        subsets = []
                        for subset_name, count in subset_counts.items():
                            proportion = count / total_section_examples
                            subset_examples = int(proportion * len(scores_ch))
                            subsets.extend([subset_name] * subset_examples)
                        
                        # Handle any remaining examples due to rounding
                        while len(subsets) < len(scores_ch):
                            subsets.append(list(subset_counts.keys())[0])
                        subsets = subsets[:len(scores_ch)]  # Trim if over
                        
                        subset_dist = {}
                        for s in subsets:
                            subset_dist[s] = subset_dist.get(s, 0) + 1
                        print(f"Used proportional reconstruction: {subset_dist}")
            else:
                # Use all subsets if no filter specified
                if len(subsets_only) == len(scores_ch):
                    subsets = subsets_only
                    print(f"Successfully reconstructed {len(subsets)} proper subset labels")
                else:
                    print(f"Length mismatch - using fallback reconstruction")
                    print(f"Original dataset: {len(subsets_only)} examples")
                    print(f"Cached scores: {len(scores_ch)} examples")
                    print(f"This suggests cached data is subset-filtered - using mixed label")
                    subsets = ["mixed"] * len(scores_ch)
                    
        except Exception as e:
            print(f"Could not load proper subset information: {e}")
            print(f"Using fallback subset reconstruction")
            logging.warning(f"Failed to load proper subset information: {e}")
            
            # Enhanced fallback: try to do proper subset reconstruction even in error case
            if args.filter_by_subset and args.filter_by_subset in SECTION_MAP:
                print(f"   Attempting enhanced fallback reconstruction for {args.filter_by_subset}")
                section_name = SECTION_MAP[args.filter_by_subset]
                expected_subsets = SUBSET_MAPPING[section_name]
                
                # Use proportional reconstruction based on official RewardBench subset counts
                from rewardbench.constants import EXAMPLE_COUNTS
                subsets = []
                total_examples = len(scores_ch)
                
                # Calculate proportions for each subset based on official counts
                section_total = sum(EXAMPLE_COUNTS.get(subset, 0) for subset in expected_subsets)
                
                for subset_name in expected_subsets:
                    subset_count = EXAMPLE_COUNTS.get(subset_name, 0)
                    if section_total > 0:
                        proportion = subset_count / section_total
                        subset_examples = int(proportion * total_examples)
                        subsets.extend([subset_name] * subset_examples)
                
                # Handle any remaining examples due to rounding
                while len(subsets) < total_examples:
                    subsets.append(expected_subsets[0])
                subsets = subsets[:total_examples]  # Trim if over
                
                subset_dist = {}
                for s in subsets:
                    subset_dist[s] = subset_dist.get(s, 0) + 1
                print(f"   ✅ Enhanced fallback reconstruction: {subset_dist}")
            else:
                # Simple fallback
                subset_name = args.filter_by_subset if args.filter_by_subset else "mixed"
                subsets = [subset_name] * len(scores_ch)
                print(f"   Using simple fallback subset '{subset_name}' for {len(scores_ch)} examples")

    subset_counts      = Counter(subsets)
    subset_correct = Counter([subsets[i] for i in correct_idx])
    print("Per-subset counts:", subset_counts)
    print("Per-subset correct:", subset_correct)
    logging.info(f"subset counts  : {subset_counts}")
    logging.info(f"subset correct : {subset_correct}")
    
    results_grouped = {}
    for subset in set(subsets):
        subset_mask = [i for i, s in enumerate(subsets) if s == subset]
        subset_correct_count = len([i for i in subset_mask if i in correct_idx])
        subset_accuracy = subset_correct_count / len(subset_mask)
        results_grouped[subset] = subset_accuracy
        print(f"Subset {subset}: {subset_correct_count}/{len(subset_mask)} = {subset_accuracy:.4f}")
    
    if args.filter_by_subset and args.filter_by_subset in SECTION_MAP:
        section_name = SECTION_MAP[args.filter_by_subset]
        print(f"Expected subsets for {section_name}: {SUBSET_MAPPING[section_name]}")
        print(f"Actual subsets found: {list(results_grouped.keys())}")
        section_score = calculate_scores_per_section(EXAMPLE_COUNTS, {section_name: SUBSET_MAPPING[section_name]}, results_grouped)
        official_score = section_score[section_name]
        print(f"Official weighted {section_name} score: {official_score:.4f} ({official_score*100:.2f}%)")
        logging.info(f"Official weighted {section_name} score: {official_score:.4f} ({official_score*100:.2f}%)")

    if args.eval_only:
        return

    # Skip feature saving if we loaded from cache
    if feat_diff is not None and h_ch is None and h_rj is None:
        print("Using cached features - skipping feature generation and saving")
        logging.info("Using cached features - skipping feature generation and saving")
    else:
        print("Generating and saving new features...")
        logging.info("Generating and saving new features")
        
        feat_diff = h_ch - h_rj
        
        subset_suffix = f"_{args.filter_by_subset}" if args.filter_by_subset else "_full"
        features_diff_file = f"features_diff{subset_suffix}.pkl"
        
        # Save everything to data directory for consistency
        data_dir = f"data/{model_name_short}_features"
        os.makedirs(data_dir, exist_ok=True)

        # Save scores to data directory with subset suffix for consistency
        with open(os.path.join(data_dir, f"scores_chosen{subset_suffix}.pkl"), "wb") as f: pkl.dump(scores_ch, f)
        with open(os.path.join(data_dir, f"scores_rejected{subset_suffix}.pkl"), "wb") as f: pkl.dump(scores_rj, f)

        # Save full features for potential future use
        with open(os.path.join(data_dir, "features_chosen_full.pkl"), "wb") as f: pkl.dump(h_ch, f)
        with open(os.path.join(data_dir, "features_rejected_full.pkl"), "wb") as f: pkl.dump(h_rj, f)
        
        # Save feature differences
        with open(os.path.join(data_dir, "features_diff_full.pkl"), "wb") as f: pkl.dump(feat_diff, f)
        with open(os.path.join(data_dir, "features_diff_correct_only.pkl"), "wb") as f:
            pkl.dump(feat_diff[correct_idx], f)
        
        # Save subset-specific features for redundancy analysis
        with open(os.path.join(data_dir, features_diff_file), "wb") as f: pkl.dump(feat_diff, f)

        print(f"Saved all artifacts to {data_dir}")
    
    # Ensure we have the feat_diff for redundancy analysis
    if feat_diff is None and h_ch is not None and h_rj is not None:
        feat_diff = h_ch - h_rj

    if args.find_redundancy:
        is_gold_standard = MODEL_ID == GOLD_STANDARD_MODEL
        
        if is_gold_standard:
            print(f"\nThis is the gold standard model ({GOLD_STANDARD_MODEL}). Running redundancy detection...")
            logging.info(f"Running redundancy detection for gold standard model: {MODEL_ID}")
            
            try:
                find_redundancy(
                    solve_by="r2", 
                    data_name=f"features_diff{subset_suffix}",
                    model_name=model_name_short, 
                    solver=args.redundancy_solver,
                    threshold=args.redundancy_threshold
                )
                print(f"Redundancy detection completed. Results saved to data/{model_name_short}_features/")
                
                redundant_pkl_path = f"data/{model_name_short}_features/redundant_features_diff{subset_suffix}_{args.redundancy_solver}_{args.redundancy_threshold}.pkl"
                
                if os.path.exists(redundant_pkl_path):
                    print(f"\nRunning redundancy tests using {redundant_pkl_path}")
                    logging.info(f"Running redundancy tests using {redundant_pkl_path}")
                    
                    try:
                        with open(redundant_pkl_path, "rb") as fh:
                            redundant_indices = pkl.load(fh)
                        
                        # Pass section name for weighted accuracy calculation
                        section_name = SECTION_MAP.get(args.filter_by_subset, None)
                        redundancy_results = run_redundancy_tests(scores_ch, scores_rj, redundant_indices, len(scores_ch), section_name)
                        print("Redundancy test results:", redundancy_results)
                        logging.info(f"Redundancy test results: {json.dumps(redundancy_results, indent=2)}")
                        
                        results_dir = os.path.join(LOGDIR, "redundancy_results")
                        os.makedirs(results_dir, exist_ok=True)
                        
                        results_filename = f"{model_name_short}_{args.filter_by_subset or 'all'}_{args.redundancy_solver}_{args.redundancy_threshold}_redundancy_results_ablation.json"
                        results_path = os.path.join(results_dir, results_filename)
                        
                        with open(results_path, 'w') as f:
                            json.dump(redundancy_results, f, indent=4)
                        
                        print(f"Redundancy test results saved to {results_path}")
                        logging.info(f"Redundancy test results saved to {results_path}")
                        
                    except Exception as e:
                        logging.error(f"Error in redundancy testing: {e}")
                        print(f"Error in redundancy testing: {e}")
                else:
                    print(f"Redundant indices file not found at {redundant_pkl_path}")
                    logging.warning(f"Redundant indices file not found at {redundant_pkl_path}")
                    
            except Exception as e:
                logging.error(f"Error in redundancy detection: {e}")
                print(f"Error in redundancy detection: {e}")
        else:
            print(f"\nNot the gold standard model. Looking for existing redundant indices from gold standard...")
            logging.info(f"Current model ({MODEL_ID}) is not gold standard ({GOLD_STANDARD_MODEL}). Looking for existing redundant indices.")
            
            # Look for gold standard redundant indices
            gold_redundant_indices, indices_found = get_gold_standard_redundant_indices(
                subset_suffix, args.redundancy_solver, args.redundancy_threshold
            )
            
            if indices_found and gold_redundant_indices is not None:
                print(f"Found gold standard redundant indices. Running ablation tests...")
                logging.info(f"Using gold standard redundant indices for ablation testing.")
                
                try:
                    # Pass section name for weighted accuracy calculation
                    section_name = SECTION_MAP.get(args.filter_by_subset, None)
                    redundancy_results = run_redundancy_tests(scores_ch, scores_rj, gold_redundant_indices, len(scores_ch), section_name)
                    print("Redundancy test results:", redundancy_results)
                    logging.info(f"Redundancy test results: {json.dumps(redundancy_results, indent=2)}")
                    
                    results_dir = os.path.join(LOGDIR, "redundancy_results")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    results_filename = f"{model_name_short}_{args.filter_by_subset or 'all'}_{args.redundancy_solver}_{args.redundancy_threshold}_redundancy_results_ablation.json"
                    results_path = os.path.join(results_dir, results_filename)
                    
                    with open(results_path, 'w') as f:
                        json.dump(redundancy_results, f, indent=4)
                    
                    print(f"Redundancy test results saved to {results_path}")
                    logging.info(f"Redundancy test results saved to {results_path}")
                    
                except Exception as e:
                    logging.error(f"Error in redundancy testing: {e}")
                    print(f"Error in redundancy testing: {e}")
            else:
                error_msg = f"Gold standard redundant indices not available for subset '{args.filter_by_subset or 'all'}' with solver '{args.redundancy_solver}' and threshold '{args.redundancy_threshold}'. Please run redundancy detection on the gold standard model first."
                print(f"\nError: {error_msg}")
                logging.error(error_msg)
                print(f"Expected path: data/{GOLD_STANDARD_MODEL.split('/')[-1]}_features/redundant_features_diff{subset_suffix}_{args.redundancy_solver}_{args.redundancy_threshold}.pkl")
                return


if __name__ == "__main__":
    main()
