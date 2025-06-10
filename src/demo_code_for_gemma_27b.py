import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = "cuda" # you can use "auto" for placing the model on several GPUs if your GPUs can not fit the model
path = "nicolinho/QRM-Gemma-2-27B"
cache_directory = "/scratch/general/vast/u1472659/huggingface_cache/"
model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device, cache_dir=cache_directory, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
# We load a random sample from the validation set of the HelpSteer dataset
prompt = 'Does pineapple belong on a Pizza?'
response = "There are different opinions on this. Some people like pineapple on a Pizza while others condemn this."
messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
with torch.no_grad():
   output = model(input_ids)
   # Expectation of the reward distribution
   reward = output.score.cpu().float() 
   # Quantile estimates for the quantiles 0.05, 0.1, ..., 0.9, 0.95 representing the distribution over rewards
   reward_quantiles = output.reward_quantiles.cpu().float()


print(reward)
print(reward_quantiles)

# The attributes of the 5 reward objectives
attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity']
