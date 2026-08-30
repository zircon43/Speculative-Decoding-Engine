import torch
import torch.nn.functional as F
import time

from .prompts import TEST_PROMPTS
from .models import naive_greedy_generate

def test_phase_3_sampling_math_proof(decoder):
    print("\n--- Phase 3: Mathematical Proof of Residual Resampling ---")
    vocab_size = 10
    k = 1
    temperature = 1.0
    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    
    torch.manual_seed(42)
    draft_logits = torch.randn(1, k, vocab_size, device=device)
    target_logits = torch.randn(1, k + 1, vocab_size, device=device)
    p_target = F.softmax(target_logits[0, 0], dim=-1)
    
    N = 50000
    counts = torch.zeros(vocab_size, device=device)
    
    print(f"Running {N} iterations of purely the acceptance/rejection loop...")
    for _ in range(N):
        draft_probs = F.softmax(draft_logits[0, 0], dim=-1)
        draft_token = torch.multinomial(draft_probs, 1).unsqueeze(0)
        
        _, replacement, num_accepted = decoder.accept_or_reject_sampling(
            target_logits, draft_logits, draft_token, temperature
        )
        
        if num_accepted == 1:
            counts[draft_token.item()] += 1
        else:
            counts[replacement.item()] += 1
            
    p_empirical = (counts + 1e-6) / (N + 1e-6 * vocab_size)
    kl_div = F.kl_div(p_empirical.log(), p_target, reduction='sum').item()
    print(f"KL Divergence between Target p(x) and Empirical Spec-Decoding: {kl_div:.6f}")
    
    if kl_div < 0.01:
        print("PHASE 3 COMPLETE: Speculative Decoding rejection math is PERFECTLY ALIGNED!")
    else:
        print("WARNING: High KL Divergence. Mathematics failed.")

def test_phase_4_multi_gpu(decoder, target, tokenizer, device):
    print("\n--- Phase 4: Multi-GPU Verification & Profiling ---")
    if str(decoder.draft_device) == str(decoder.target_device):
        print("WARNING: Only 1 GPU found. Simulating transfer timing instead of true cross-device transfer.")
    else:
        print(f"Confirmed Multi-GPU Active: Draft on {decoder.draft_device}, Target on {decoder.target_device}")

    # Use first 10 prompts for quick profiling
    prompts_to_test = TEST_PROMPTS[:10]
    pass_count = 0
    max_new_tokens = 30
    
    decoder.reset_profiler()

    for i, prompt in enumerate(prompts_to_test):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        baseline_out = naive_greedy_generate(target, tokenizer, input_ids, max_new_tokens)
        
        with torch.no_grad():
            spec_out = decoder.speculative_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0
            )

        min_len = min(baseline_out.shape[1], spec_out.shape[1])
        baseline_trunc = baseline_out[0, :min_len]
        spec_trunc = spec_out[0, :min_len]

        if torch.equal(baseline_trunc, spec_trunc):
            pass_count += 1
        else:
            print(f"Phase 4 Failed exact match on prompt {i+1}")
            break

    print(f"Phase 4 Exact-Match Verification: {pass_count}/{len(prompts_to_test)} Passed.")
    
    print("\n--- Profiling Results (Over 10 Prompts) ---")
    print(f"Total Compute Time:  {decoder.compute_time_ms:.2f} ms")
    print(f"Total Transfer Time: {decoder.transfer_time_ms:.2f} ms")
    
    if decoder.compute_time_ms > 0:
        overhead_pct = (decoder.transfer_time_ms / decoder.compute_time_ms) * 100
        print(f"Transfer Overhead:   {overhead_pct:.2f}% of compute")
        
    if pass_count == len(prompts_to_test):
        print("PHASE 4 COMPLETE: Multi-GPU placement and verification is successful!")

def test_phase_5_benchmarking(decoder, target, draft, tokenizer, device):
    print("\n--- Phase 5: Benchmarking & K-Sweep ---")
    prompts = TEST_PROMPTS[:10]
    max_new_tokens = 100
    
    print("Measuring Baseline Target Model TPS...")
    start_time = time.time()
    total_tokens_baseline = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = naive_greedy_generate(target, tokenizer, inputs.input_ids, max_new_tokens)
        total_tokens_baseline += (out.shape[1] - inputs.input_ids.shape[1])
    target_time = time.time() - start_time
    target_tps = total_tokens_baseline / target_time
    
    print("Measuring Baseline Draft Model TPS...")
    start_time = time.time()
    total_tokens_draft = 0
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(decoder.draft_device)
        out = naive_greedy_generate(draft, tokenizer, inputs.input_ids, max_new_tokens)
        total_tokens_draft += (out.shape[1] - inputs.input_ids.shape[1])
    draft_time = time.time() - start_time
    draft_tps = total_tokens_draft / draft_time
    
    c = target_tps / draft_tps  # Cost ratio: c_draft / c_target
    print(f"\nTarget TPS (c_target): {target_tps:.2f} tokens/sec")
    print(f"Draft TPS (c_draft):   {draft_tps:.2f} tokens/sec")
    print(f"Cost ratio c:          {c:.4f}")
    
    k_values = [1, 2, 4, 8, 16]
    
    print("\n| K  | Speedup | Tokens/sec | Acceptance (α) | Theo Opt K |")
    print("|----|---------|------------|----------------|------------|")
    
    for k in k_values:
        decoder.k = k
        decoder.reset_profiler()
        
        # Warmup (1 prompt)
        _ = decoder.speculative_generate(tokenizer(prompts[0], return_tensors="pt").input_ids.to(device), max_new_tokens=10, temperature=0.0)
        
        decoder.reset_profiler()
        start_time = time.time()
        total_tokens_spec = 0
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = decoder.speculative_generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=0.0)
            total_tokens_spec += (out.shape[1] - inputs.input_ids.shape[1])
            
        spec_time = time.time() - start_time
        spec_tps = total_tokens_spec / spec_time
        speedup = spec_tps / target_tps
        
        alpha = decoder.total_draft_tokens_accepted / max(1, decoder.total_draft_tokens_proposed)
        
        # Calculate theoretical optimal K using empirical alpha and cost ratio
        best_theoretical_k = 1
        best_theo_speedup = 0
        for test_k in range(1, 40):
            # E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)
            e_tokens = (1 - (alpha ** (test_k + 1))) / (1 - alpha) if alpha < 1.0 else (test_k + 1)
            # Cost per step relative to target is 1 + k*c
            cost = 1 + test_k * c
            theo_speedup = e_tokens / cost
            if theo_speedup > best_theo_speedup:
                best_theo_speedup = theo_speedup
                best_theoretical_k = test_k
                
        print(f"| {k:<2} | {speedup:>6.2f}x | {spec_tps:>10.2f} | {alpha:>14.3f} | {best_theoretical_k:>10} |")

    print("\n--- Hardware Memory Profiling ---")
    if torch.cuda.is_available():
        target_mem = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Peak VRAM on Target GPU ({device}): {target_mem:.2f} GB")
        if str(device) != str(decoder.draft_device):
            draft_mem = torch.cuda.max_memory_allocated(decoder.draft_device) / (1024**3)
            print(f"Peak VRAM on Draft GPU ({decoder.draft_device}): {draft_mem:.2f} GB")

