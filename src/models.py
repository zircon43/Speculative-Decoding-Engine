import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_models_and_tokenizer(
    draft_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    target_model_id="Qwen/Qwen2.5-7B-Instruct",
    target_device="cuda:0",
    draft_device="cuda:1" # Phase 4 target
):
    dtype = torch.float16 # Required to fit 7B on Kaggle 16GB T4 GPUs

    # Fallback to single GPU for local syntax testing if 2 GPUs aren't available
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Less than 2 GPUs found. Defaulting both models to cuda:0 for safety.")
        draft_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        target_device = draft_device

    print(f"Loading tokenizer from {target_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    
    print(f"Loading tokenizer from {draft_model_id} for verification...")
    draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
    
    print("Asserting tokenizer vocabularies are identical...")
    assert draft_tokenizer.get_vocab() == tokenizer.get_vocab(), \
        "CRITICAL ERROR: Draft and target models do not share the exact same tokenizer vocabulary!"
        
    print(f"Loading draft model ({draft_model_id}) on {draft_device} in {dtype}...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=dtype,
        device_map=draft_device
    )
    draft_model.eval()

    print(f"Loading target model ({target_model_id}) on {target_device} in {dtype}...")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map=target_device
    )
    target_model.eval()
    
    return draft_model, target_model, tokenizer

def naive_greedy_generate(model, tokenizer, input_ids, max_new_tokens):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past = outputs.past_key_values
        next_token = torch.argmax(
            outputs.logits[:, -1, :].float(), dim=-1
        ).unsqueeze(-1)

        generated = [input_ids, next_token]

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token = torch.argmax(
                outputs.logits[:, -1, :].float(), dim=-1
            ).unsqueeze(-1)
            generated.append(next_token)

            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

    return torch.cat(generated, dim=1)

