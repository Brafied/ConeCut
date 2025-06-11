# alignment_benchmark_LLM
exploring redundancy removal and alignment for LLM reward models

Idea is to see if any test examples can be removed because they do not add any new half space constraints. If yes, does the performance drop, increase or decrease on the test set after the removal. 


This employs a linear combination test that approximates redundancy in datatsets. 

To run the cone membership test for redundancy
python remove_redundancy_removing_non_red.py --data_name safety --model_name skyworks_llama 