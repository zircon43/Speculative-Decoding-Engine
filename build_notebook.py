import json

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def create_code_cell(source):
    # Fix source list formatting (ipynb expects a list of lines with trailing newlines)
    lines = source.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]]
    if len(lines) > 0:
        source_lines.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }
    
def create_md_cell(source):
    lines = source.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]]
    if len(lines) > 0:
        source_lines.append(lines[-1])
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def clean_imports(text):
    """Strip out standard local and global imports so they don't repeat in every cell."""
    out = []
    for line in text.split('\n'):
        if line.startswith('import torch') or line.startswith('import time') or line.startswith('from transformers') or line.startswith('from .'):
            continue
        out.append(line)
    return '\n'.join(out).strip()

def main():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # 1. Setup
    notebook['cells'].append(create_md_cell("# 1. Installation & Imports\nThis cell sets up the environment for Kaggle T4 dual-GPU speculative decoding."))
    notebook['cells'].append(create_code_cell("!pip install transformers accelerate\n\nimport torch\nimport torch.nn.functional as F\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\nimport time"))

    # 2. Prompts
    notebook['cells'].append(create_md_cell("# 2. Prompts Dataset\nOur 50-prompt test suite spanning factual, long-form, code, and chat tasks."))
    notebook['cells'].append(create_code_cell(clean_imports(read_file('src/prompts.py'))))

    # 3. Models
    notebook['cells'].append(create_md_cell("# 3. Model Loading & Baseline Generator\nFunctions to load the Target (7B) and Draft (0.5B) models in FP16 on separate GPUs, plus the standard autoregressive baseline."))
    notebook['cells'].append(create_code_cell(clean_imports(read_file('src/models.py'))))

    # 4. Decoder Engine
    notebook['cells'].append(create_md_cell("# 4. The Speculative Decoding Engine\nThe core engine implementing KV-cache rollback, log-space acceptance, and residual resampling mathematically."))
    notebook['cells'].append(create_code_cell(clean_imports(read_file('src/decoder.py'))))

    # 5. Benchmarking
    notebook['cells'].append(create_md_cell("# 5. Verification & Benchmarking Harness\nTest suite for Phase 3 (Math Proof), Phase 4 (Multi-GPU Profiling), and Phase 5 (K-Sweep Benchmarking)."))
    notebook['cells'].append(create_code_cell(clean_imports(read_file('src/benchmark.py'))))

    # 6. Execution
    notebook['cells'].append(create_md_cell("# 6. Execution Main\nRun this cell to fire off the full pipeline!"))
    
    # We replace the local main block in benchmark.py with a straight run since we stripped imports
    exec_block = """print("--- Phase 0: Scaffolding ---")
draft, target, tokenizer = load_models_and_tokenizer(
    draft_model_id="Qwen/Qwen2.5-0.5B-Instruct",
    target_model_id="Qwen/Qwen2.5-7B-Instruct",
)
device = next(target.parameters()).device
decoder = SpeculativeDecoder(draft, target, tokenizer, k=4)
max_new_tokens = 30
pass_count = 0
total = len(TEST_PROMPTS)

print(f"\\n--- Phase 1 & 2: Correctness Test ({total} Prompts) ---")
print(f"Baseline: naive greedy (no logits processors)")
print(f"Test:     speculative decoder (k={decoder.k})")

for i, prompt in enumerate(TEST_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    baseline_out = naive_greedy_generate(
        target, tokenizer, input_ids, max_new_tokens
    )
    with torch.no_grad():
        spec_out = decoder.speculative_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0
        )

    min_len = min(baseline_out.shape[1], spec_out.shape[1])
    baseline_trunc = baseline_out[0, :min_len]
    spec_trunc = spec_out[0, :min_len]

    match = torch.equal(baseline_trunc, spec_trunc)

    if match:
        pass_count += 1
    else:
        print(f"Prompt {i+1}/{total} [FAIL]")
        break

print(f"\\nResult: {pass_count}/{total} Passed.")
if pass_count == total:
    print("PHASE 2 COMPLETE: Speculative Decoder is 100% Greedy Exact-Match Correct!")
    
test_phase_3_sampling_math_proof(decoder)
test_phase_4_multi_gpu(decoder, target, tokenizer, device)
test_phase_5_benchmarking(decoder, target, draft, tokenizer, device)
"""
    notebook['cells'].append(create_code_cell(exec_block))

    with open('kaggle_spec_decode.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == '__main__':
    main()
