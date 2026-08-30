import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    def __init__(self, draft_model, target_model, tokenizer, k=4):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.k = k
        self.draft_device = next(draft_model.parameters()).device
        self.target_device = next(target_model.parameters()).device
        
        # Profiling stats
        self.transfer_time_ms = 0.0
        self.compute_time_ms = 0.0
        
        # Alpha tracking stats
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0
        
    def reset_profiler(self):
        self.transfer_time_ms = 0.0
        self.compute_time_ms = 0.0
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0

    def _transfer(self, tensor, device):
        """Cross-device tensor transfer with strict event isolation."""
        if str(tensor.device) == str(device):
            return tensor
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = tensor.to(device)
        end.record()
        torch.cuda.synchronize()
        self.transfer_time_ms += start.elapsed_time(end)
        return res
        
    def _run_compute(self, fn, *args, **kwargs):
        """Wraps compute functions to track compute isolation time."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        res = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self.compute_time_ms += start.elapsed_time(end)
        return res

    def truncate_kv_cache(self, past_key_values, keep_length):
        if past_key_values is None:
            return None
        past_key_values.crop(keep_length)
        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(past_key_values, attr):
                current = getattr(past_key_values, attr)
                if isinstance(current, int) and current != keep_length:
                    setattr(past_key_values, attr, keep_length)
        actual = past_key_values.get_seq_length()
        assert actual == keep_length, (
            f"After crop({keep_length}), get_seq_length()={actual}. "
            f"Internal state: {past_key_values.__dict__}"
        )
        return past_key_values

    def draft_k_tokens(self, input_ids, past_key_values, temperature=0.0):
        draft_tokens = []
        draft_logits_list = []
        current_input_ids = input_ids

        for _ in range(self.k):
            with torch.no_grad():
                outputs = self.draft_model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits_f32 = outputs.logits[:, -1, :].float()
            draft_logits_list.append(logits_f32)
            
            if temperature == 0.0:
                next_token_id = torch.argmax(logits_f32, dim=-1).unsqueeze(-1)
            else:
                probs = F.softmax(logits_f32 / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
            draft_tokens.append(next_token_id)
            current_input_ids = next_token_id

        return (
            torch.cat(draft_tokens, dim=1),
            torch.stack(draft_logits_list, dim=1),
            past_key_values,
        )

    def verify_with_target(self, input_ids, past_key_values):
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return outputs.logits.float(), outputs.past_key_values

    def accept_or_reject_greedy(self, target_logits, draft_tokens):
        batch_size, k = draft_tokens.shape
        assert batch_size == 1, "Only batch_size=1 supported for now"

        target_predictions = torch.argmax(target_logits, dim=-1)

        accepted_tokens = []
        num_accepted = 0

        for i in range(k):
            if draft_tokens[0, i].item() == target_predictions[0, i].item():
                accepted_tokens.append(draft_tokens[0, i].item())
                num_accepted += 1
            else:
                break

        replacement_token = target_predictions[:, num_accepted].unsqueeze(-1)

        if accepted_tokens:
            accepted_tensor = torch.tensor(
                accepted_tokens, device=draft_tokens.device
            ).unsqueeze(0)
        else:
            accepted_tensor = torch.empty(
                (1, 0), dtype=torch.long, device=draft_tokens.device
            )

        return accepted_tensor, replacement_token, num_accepted

    def accept_or_reject_sampling(self, target_logits, draft_logits, draft_tokens, temperature):
        batch_size, k = draft_tokens.shape
        assert batch_size == 1, "Only batch_size=1 supported for now"

        target_log_probs = F.log_softmax(target_logits / temperature, dim=-1)
        draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)

        accepted_tokens = []
        num_accepted = 0

        for i in range(k):
            token = draft_tokens[0, i].item()
            log_p_target = target_log_probs[0, i, token]
            log_p_draft = draft_log_probs[0, i, token]

            log_ratio = log_p_target - log_p_draft
            r = torch.rand(1, device=draft_tokens.device)
            log_r = torch.log(r).item()

            if log_r < log_ratio.item():
                accepted_tokens.append(token)
                num_accepted += 1
            else:
                break

        if num_accepted == k:
            bonus_probs = F.softmax(target_logits[:, k] / temperature, dim=-1)
            replacement_token = torch.multinomial(bonus_probs, num_samples=1)
        else:
            p_target = F.softmax(target_logits[:, num_accepted] / temperature, dim=-1)
            p_draft = F.softmax(draft_logits[:, num_accepted] / temperature, dim=-1)

            res = torch.clamp(p_target - p_draft, min=0.0)
            res_sum = res.sum(dim=-1, keepdim=True)

            if res_sum.item() > 1e-8:
                res_probs = res / res_sum
            else:
                res_probs = p_target

            replacement_token = torch.multinomial(res_probs, num_samples=1)

        if accepted_tokens:
            accepted_tensor = torch.tensor(
                accepted_tokens, device=draft_tokens.device
            ).unsqueeze(0)
        else:
            accepted_tensor = torch.empty(
                (1, 0), dtype=torch.long, device=draft_tokens.device
            )

        return accepted_tensor, replacement_token, num_accepted
        
    def _draft_step_compute(self, next_token, draft_past, temperature):
        """Isolated draft compute step."""
        with torch.no_grad():
            return self.draft_k_tokens(next_token, draft_past, temperature=temperature)
            
    def _target_step_compute(self, target_input, target_past):
        """Isolated target compute step."""
        with torch.no_grad():
            return self.verify_with_target(target_input, target_past)

    def speculative_generate(self, prompt_input_ids, max_new_tokens=50, temperature=0.0):
        current_input_ids = prompt_input_ids
        
        # Prefill: Pre-transfer inputs so transfer time is isolated from compute time
        draft_inputs = self._transfer(current_input_ids, self.draft_device)
        target_inputs = self._transfer(current_input_ids, self.target_device)
        
        start_compute = torch.cuda.Event(enable_timing=True)
        end_compute = torch.cuda.Event(enable_timing=True)
        
        start_compute.record()
        with torch.no_grad():
            draft_outputs = self.draft_model(input_ids=draft_inputs, use_cache=True)
            draft_past = draft_outputs.past_key_values

            target_outputs = self.target_model(input_ids=target_inputs, use_cache=True)
            target_past = target_outputs.past_key_values
            
            target_logits = target_outputs.logits[:, -1, :].float()
            if temperature == 0.0:
                next_token = torch.argmax(target_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(target_logits / temperature, dim=-1), num_samples=1)

        end_compute.record()
        torch.cuda.synchronize()
        self.compute_time_ms += start_compute.elapsed_time(end_compute)

        generated_ids = [self._transfer(current_input_ids, self.target_device), next_token]
        num_generated = 1
        cache_len = current_input_ids.shape[1]

        while num_generated < max_new_tokens:
            # Transfer next token to draft
            next_token_draft = self._transfer(next_token, self.draft_device)
            
            # Draft K tokens
            draft_tokens, draft_logits, new_draft_past = self._run_compute(
                self._draft_step_compute, next_token_draft, draft_past, temperature
            )
            
            # Transfer drafts back to target
            draft_tokens_target = self._transfer(draft_tokens, self.target_device)
            target_input = torch.cat([next_token, draft_tokens_target], dim=1)
            
            # Target verification
            target_logits, new_target_past = self._run_compute(
                self._target_step_compute, target_input, target_past
            )
            
            # Transfer logits if sampling
            if temperature > 0.0:
                draft_logits_target = self._transfer(draft_logits, self.target_device)
            
            # Acceptance logic (purely on target device compute)
            start_compute.record()
            if temperature == 0.0:
                accepted_tokens, replacement_token, num_accepted = (
                    self.accept_or_reject_greedy(target_logits, draft_tokens_target)
                )
            else:
                accepted_tokens, replacement_token, num_accepted = (
                    self.accept_or_reject_sampling(target_logits, draft_logits_target, draft_tokens_target, temperature)
                )

            # Track stats
            self.total_draft_tokens_proposed += self.k
            self.total_draft_tokens_accepted += num_accepted

            if num_accepted > 0:
                generated_ids.append(accepted_tokens)
            generated_ids.append(replacement_token)
            num_generated += num_accepted + 1
            
            end_compute.record()
            torch.cuda.synchronize()
            self.compute_time_ms += start_compute.elapsed_time(end_compute)

            # KV cache rollback - Pre-transfer bonus token if all accepted
            if num_accepted == self.k:
                draft_bonus_input = self._transfer(draft_tokens_target[:, -1:], self.draft_device)
                
                start_compute.record()
                with torch.no_grad():
                    extra = self.draft_model(
                        input_ids=draft_bonus_input,
                        past_key_values=new_draft_past,
                        use_cache=True,
                    )
                new_draft_past = extra.past_key_values
                end_compute.record()
                torch.cuda.synchronize()
                self.compute_time_ms += start_compute.elapsed_time(end_compute)

            start_compute.record()
            keep_len = cache_len + 1 + num_accepted
            draft_past = self.truncate_kv_cache(new_draft_past, keep_len)
            target_past = self.truncate_kv_cache(new_target_past, keep_len)

            cache_len = keep_len
            next_token = replacement_token

            end_compute.record()
            torch.cuda.synchronize()
            self.compute_time_ms += start_compute.elapsed_time(end_compute)

            if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                break

        return torch.cat(generated_ids, dim=1)

