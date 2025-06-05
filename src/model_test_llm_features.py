import os

os.environ["HF_HOME"] = "/scratch/general/vast/u1472659/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/general/vast/u1472659/huggingface_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/scratch/general/vast/u1472659/huggingface_cache/datasets"

cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"

import json
import glob
import torch
import logging
import argparse
import numpy as np
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm
from ldlreward import LDLRewardModel27B



from peft import PeftModel
from peft import PeftConfig
from datasets import load_dataset
from  safetensors import safe_open
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from reward_model_inference_utils import run_redundancy_tests, process_examples, calculate_accuracy, process_examples_gemma


filter_subsets_dict = {'chat': ['alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-medium'],
                        'chat_hard': [ 'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
                        'safety': ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'do not answer'],
                        'reasoning': ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust']}

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process reward bench dataset with optional hard subset filtering.")
    parser.add_argument("--filter_by_subset", default="chat_hard", help="Filter for hard subsets in the reward bench dataset.")
    #parser.add_argument("--chat", action="store_true", help="Filter for regular/easy subsets in the reward bench dataset.")
    parser.add_argument("--shorten_size", default="768", help="Take full feature length or only the first 768 dimensions")  
    parser.add_argument("--using_peft", action='store_true', help='If set, use a fine-tuned PEFT model, otherwise use the base model')
    parser.add_argument("--eval_only", action='store_true', help='If set, only get the model accuracy, and not store the data')
    parser.add_argument("--redundant_pkl_path", type=str, default="features_without_peft/Skywork-Reward-Llama-3.1-8B-v0.2/redundant_safety.pkl", help='Path to the pickle file containing list of redundant examples')
    parser.add_argument("--test_redundant", action='store_true', help='If set, perform redundancy tests')

    return parser.parse_args()


def filter_dataset(dataset, filter_by_subset):

    """Filter the dataset based on the subset field if chat_hard is enabled."""
    if filter_by_subset!= '':
        include_only = filter_subsets_dict[filter_by_subset]
        '''
        [
            'mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor',
            'llmbar-adver-GPTInst', 'llmbar-adver-GPTOut', 'llmbar-adver-manual'
        ]''' 

        # Filter dataset based on the 'subset' field
        filtered_dataset = [example for example in dataset if example['subset'] in include_only]
        logging.info("the length of chat hard is {}".format(len(filtered_dataset)))
        
        return filtered_dataset

    return dataset

def process_deberta_examples(model, tokenizer, dataset, device, args):
    """Process dataset examples and compute features and scores for chosen and rejected completions."""
    
    features_chosen = []
    features_chosen_full_length = []
    features_rejected = []
    features_rejected_full_length = []
    features_diff = []
    chosen_scores = []
    rejected_scores = []

    for example in tqdm(dataset, desc="Processing examples for DeBERTa"):
        chosen_completion = example['chosen']
        rejected_completion = example['rejected']

        # Tokenize the chosen and rejected completions
        chosen_inputs = tokenizer(chosen_completion, return_tensors="pt", truncation=True, padding=True).to(device)
        rejected_inputs = tokenizer(rejected_completion, return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):  # Mixed precision for faster computation
                chosen_output = model(**chosen_inputs, output_hidden_states=True)
                rejected_output = model(**rejected_inputs, output_hidden_states=True)

                # Get logits (reward scores)
                score_chosen = chosen_output.logits[0][0].item()
                score_rejected = rejected_output.logits[0][0].item()

                chosen_scores.append(score_chosen)
                rejected_scores.append(score_rejected)

                # Get the last hidden state representations
                hidden_states_chosen = chosen_output.hidden_states[-1]
                hidden_states_rejected = rejected_output.hidden_states[-1]

                # Extract the last token representation for each completion (CLS-like token)
                cls_embedding_chosen = hidden_states_chosen[:, 0, :].cpu().squeeze()
                cls_embedding_rejected = hidden_states_rejected[:, 0, :].cpu().squeeze()
              
                features_chosen.append(cls_embedding_chosen.cpu())
                features_rejected.append(cls_embedding_rejected.cpu())
                features_diff.append((cls_embedding_chosen - cls_embedding_rejected).cpu())


    return torch.stack(features_chosen), torch.stack(features_rejected), \
           torch.stack(features_chosen), torch.stack(features_rejected), \
           torch.stack(features_diff), chosen_scores, rejected_scores

def main():
    args = parse_args()
    # model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    model_name = "nicolinho/QRM-Gemma-2-27B"
    # model_name = "ShikaiChen/LDL-Reward-Gemma-2-27B-v0.1"
    # model_name= "OpenAssistant/reward-model-deberta-v3-large-v2"
    # model_name = "/scratch/general/vast/u1472659/lora_llama_ft/merged_model/"
    # tokenizer_name = 'Skywork/Skywork-Reward-Llama-3.1-8B'
    peft_name = '/scratch/general/vast/u1472659/lora_llama_ft/Skywork-Reward-Llama-3.1-8B-v0.2_BT_RM_len512_lora32_1e-05_dataSkywork-Reward-Preference-80K-v0.2/'
    # logging_directory_path = "~/logging/"
    logging_directory_path = "~/alignment_benchmark_LLM/logging/"
    os.makedirs(logging_directory_path, exist_ok=True)
    model_name_short = model_name.split('/')[-1]

    logging.basicConfig(filename=os.path.join(logging_directory_path, f"{model_name_short}_for_{args.filter_by_subset}_{args.shorten_size}.txt"), filemode='w',
                        level=logging.INFO)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset('allenai/reward-bench', split='filtered')

    # Filter dataset if chat_hard is enabled
    filtered_dataset = filter_dataset(dataset, args.filter_by_subset)

    # Only load the model and tokenizer after filtering the dataset
    print("Loading model and tokenizer...")

    if 'QRM-Gemma' in model_name:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device, cache_dir=cache_directory, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=cache_directory)
    elif 'LDL-Reward-Gemma-2-27B' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LDLRewardModel27B.from_pretrained(model_name, device_map="auto", cache_dir = cache_directory)

    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=cache_directory,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)


    get_modified_model = False
    print(args)
    if args.using_peft:
    # if 'freeze' in script_args.peft_name or script_args.freeze_pretrained:
        print('loading freeze nonlinear parameters')
        tensors = {}
        path_list = glob.glob(os.path.join(peft_name, "adapter-*.safetensors"))
        
        for path in path_list:
            with safe_open(path, framework="pt", device=0) as f:
                for k in f.keys():
                    if 'score' in k:
                        tensors[k] = f.get_tensor(k)

        # use the same structure as the training
        mlp_layer = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512, dtype=torch.float16),  
            nn.ReLU(),
            nn.Linear(512, 1, dtype=torch.float16)  
        )

        mlp_layer.to(device)
        # Replacing the classifier with the MLP, I've set the output layer size to be 512
        model.score = mlp_layer
        model.load_state_dict(tensors, strict=False)
        
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        # If there is lora for loading
        if os.path.exists(peft_name):
            print("yes this is peft model")
            model = PeftModel.from_pretrained(model, peft_name)
        if hasattr(model, 'merge_and_unload'):
            print("going to merge")
            model = model.merge_and_unload()


        print("we have peft merged model")

    

    # peft_config = PeftConfig.from_pretrained(peft_name)

    # peft_config.init_lora_weights = True

    # model.add_adapter(peft_config)
    # model.enable_adapters()


    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id

    # # If there is lora for loading
    # if len(peft_name) and os.path.exists(peft_name):
    #     model = PeftModel.from_pretrained(model, PeftModel)
    # if hasattr(model, 'merge_and_unload'):

    #     model = model.merge_and_unload()

    
    model.to(device)
    logging.info("Model downloaded and cached")
    logging.info(model)


    if 'deberta' in model_name:
        features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length, \
        features_diff, chosen_scores, rejected_scores = process_deberta_examples(model, tokenizer, filtered_dataset, device, args)
    elif 'Gemma' in model_name:
        features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length, \
        features_diff, chosen_scores, rejected_scores = process_examples_gemma(model, tokenizer, filtered_dataset, device, args)
    else:    
        # Process the examples
        features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length, \
        features_diff, chosen_scores, rejected_scores = process_examples(model, tokenizer, filtered_dataset, device, args)

    print("shape of features difference is {}".format(len(features_diff)))
    logging.info("shape of features difference is {}".format(len(features_diff)))

    accuracy, correct_count, chosen_np_arr, rejected_np_arr, correct_indices = calculate_accuracy(
        chosen_scores, rejected_scores)
    
    logging.info(f"Total {correct_count} datapoints where we get correct predictions out of {len(chosen_scores)}")
    logging.info(f"The percentage where score of chosen is greater than rejected is {accuracy:.2f}%")
    print(f"The percentage where score of chosen is greater than rejected is {accuracy:.2f}%")

    # ###check the 87.4% accuracy for skyworks model.
    # chosen_np_arr = np.array(chosen_scores)
    # rejected_np_arr = np.array(rejected_scores)
    # comparison = chosen_np_arr > rejected_np_arr
    # # Calculate the percentage where list A's elements are greater
    # percentage = np.mean(comparison) * 100
    # indices_of_greater = np.where(comparison)[0]

    # logging.info("total {} datapoints where we get correct predictions ".format(len(indices_of_greater)))

    # logging.info("the percentage where score of chosen is greater than rej is {}".format(percentage))
    # print("the percentage where score of chosen is greater than rej is {}".format(percentage))
    # logging.info(f" the feature size is {features_chosen.shape}")

    args.full_dataset_accuracy = accuracy

    if args.test_redundant and not args.eval_only:
        error_message = "Redundancy testing must be run with --eval_only flag to prevent unnecessarily saving features"
        logging.error(error_message)
        raise ValueError(error_message)

    if args.test_redundant and args.redundant_pkl_path:
        try:
            with open(args.redundant_pkl_path, 'rb') as f:
                redundant_examples = pkl.load(f)
                
            logging.info(f"Loaded {len(redundant_examples)} redundant examples from {args.redundant_pkl_path}")
            
            redundancy_results = run_redundancy_tests(chosen_scores, rejected_scores, redundant_examples, len(filtered_dataset))
            
            # Save redundancy test results
            results_dir = os.path.join(logging_directory_path, "redundancy_results")
            os.makedirs(results_dir, exist_ok=True)

            print(f"writing results from redundancy tests to {results_dir}/{model_name_short}/{args.filter_by_subset}")
            print(redundancy_results)
            if "nnls" in args.redundant_pkl_path:

                with open(os.path.join(results_dir, 
                                        f"{model_name_short}_{args.filter_by_subset}_nnls_redundancy_results_abelation.json"), 'w') as f:
                    json.dump(redundancy_results, f, indent = 4)
            elif "lstsq" in args.redundant_pkl_path :
                with open(os.path.join(results_dir, 
                                        f"{model_name_short}_{args.filter_by_subset}_lstsq_redundancy_results_abelation.json"), 'w') as f:
                    json.dump(redundancy_results, f, indent = 4)
            else:
                with open(os.path.join(results_dir, 
                                        f"{model_name_short}_{args.filter_by_subset}_redundancy_results_abelation.json"), 'w') as f:
                    json.dump(redundancy_results, f, indent = 4)
                
        except Exception as e:
            logging.error(f"Error in redundancy testing: {str(e)}")
            print(f"Error in redundancy testing: {str(e)}")


    if args.eval_only:
        exit()


    if args.using_peft:
        base_directory_for_features = "features_with_peft"

    else:
        base_directory_for_features = "features_without_peft"
    model_directory = os.path.join(base_directory_for_features, model_name_short)
    os.makedirs(model_directory, exist_ok=True)


    print("the model directory is {}".format(model_directory))


    with open(os.path.join(model_directory, 'scores_chosen.pkl'), 'wb') as f:
        pkl.dump(chosen_np_arr, f)

    with open(os.path.join(model_directory, 'scores_rejected.pkl'), 'wb') as f:
        pkl.dump(rejected_np_arr, f)
    
    # Save features to a file
    logging.info(f"Shape of features chosen is : {features_chosen.shape}")
    logging.info(f"Shape of features rejection is : {features_rejected.shape}")

    for save_name, features_var in zip(['features_chosen', 'features_rejected', 'features_chosen_full_length', 'features_rejected_full_length'], [features_chosen, features_rejected, features_chosen_full_length, features_rejected_full_length]):

        if len(features_var) > 0:
            if 'full_length' in save_name:
                features_size = '' #no suffix neaded, default is 4096
            else:
                features_size = str(args.shorten_size)

            with open(os.path.join(model_directory, '{}_{}{}.pkl'.format(save_name, args.filter_by_subset, features_size)), 'wb') as f:
                pkl.dump(features_var, f)

    if len(features_diff) > 0:

        logging.info("here writing difference, {}".format(features_diff.shape))

        print("writing difference in features")
        with open(os.path.join(model_directory, 'features_diff_full_length_{}.pkl'.format(args.filter_by_subset)), 'wb') as f:
            pkl.dump(features_diff, f)

        print("writing only accurate fetaure differences")
        with open(os.path.join(model_directory, 'features_diff_correct_only_full_length_{}.pkl'.format(args.filter_by_subset)), 'wb') as f1:
            pkl.dump(features_diff[correct_indices], f1)


if __name__ == "__main__":


    main()

