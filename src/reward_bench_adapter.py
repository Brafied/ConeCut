from rewardbench import load_eval_dataset, REWARD_MODEL_CONFIG, check_tokenizer_chat_template
from rewardbench.constants import SUBSET_MAPPING
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, math, random, inspect

def build_rm(model_id: str,
             torch_dtype=torch.bfloat16,
             quantize_toggle: bool | None = None,
             attn_impl: str | None = None,
             cache_dir: str | None = None):

    cfg = REWARD_MODEL_CONFIG.get(model_id, REWARD_MODEL_CONFIG["default"])
    model_builder     = cfg["model_builder"]
    quantized_default = cfg["quantized"]
    model_kwargs = {}

    if quantize_toggle is None:
        quantized = quantized_default
    else:
        quantized = quantize_toggle

    # auto-disable quant for bfloat16 (mimic run_rm.py)
    if torch_dtype == torch.bfloat16:
        quantized = False

    if quantized:
        model_kwargs.update(
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    else:
        model_kwargs.update(
            device_map="auto",
            torch_dtype=torch_dtype,
        )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model  = model_builder(model_id, **model_kwargs, trust_remote_code=False, cache_dir=cache_dir)
    tok    = AutoTokenizer.from_pretrained(model_id)
    return tok, model

def load_subset(tokenizer,
                section: str | None = None,
                keep_indices: list[int] | None = None):
    ds, subsets = load_eval_dataset(
        core_set=True,
        tokenizer=tokenizer,
        logger=None,
        return_extra_data=False,
    )

    if section:
        wanted = set(SUBSET_MAPPING[section])
        mask   = [i for i,s in enumerate(subsets) if s in wanted]
        ds, subsets = ds.select(mask), [subsets[i] for i in mask]

    if keep_indices is not None:
        ds, subsets = ds.select(keep_indices), [subsets[i] for i in keep_indices]

    return ds, subsets

def forward_collect(model, tokenizer, texts, device, max_length=2048):
    inputs = tokenizer(texts,
                       return_tensors="pt",
                       padding=True, truncation=True,
                       max_length=max_length).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    logits  = out.logits.float().squeeze(-1).cpu()
    h_last  = out.hidden_states[-1][:,0,:].cpu()   # CLS / BOS embedding
    return logits, h_last


def run_rm_subset(model_id: str,
                  section: str | None = None,
                  keep_indices: list[int] | None = None,
                  batch_size: int = 8,
                  torch_dtype=torch.bfloat16,
                  max_length: int = 2048,
                  cache_dir: str | None = None):

    tok, rm = build_rm(model_id, torch_dtype=torch_dtype, cache_dir=cache_dir)
    device  = next(rm.parameters()).device


    ##paddings and truncation from run_rm.py file

    tok.padding_side = "left"
    tok.truncation_side = "left"
    if not check_tokenizer_chat_template(tok):
        tok.add_eos_token = True
    if rm.config.pad_token_id is None:
        rm.config.pad_token_id = tok.eos_token_id
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    ds, subsets = load_subset(tok, section, keep_indices)

    scores_ch, scores_rj = [], []
    hid_ch, hid_rj       = [], []

    for i in range(0, len(ds), batch_size):
        batch = ds[i : i + batch_size]
        logits_ch, hidden_ch = forward_collect(
            rm, tok, batch["text_chosen"], device, max_length)
        logits_rj, hidden_rj = forward_collect(
            rm, tok, batch["text_rejected"], device, max_length)

        scores_ch.extend(logits_ch.tolist())
        scores_rj.extend(logits_rj.tolist())
        hid_ch.append(hidden_ch)
        hid_rj.append(hidden_rj)

    hid_ch = torch.cat(hid_ch)
    hid_rj = torch.cat(hid_rj)
    return scores_ch, scores_rj, hid_ch, hid_rj, subsets
