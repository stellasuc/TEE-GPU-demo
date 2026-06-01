from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    F = None
    nn = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class MaskedOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_masked_linear_matches_torch_linear(self) -> None:
        from tee_gpu_demo.masked_ops import masked_linear

        x = torch.randn(3, 5)
        weight = torch.randn(7, 5)
        bias = torch.randn(7)

        expected = F.linear(x, weight, bias)
        actual = masked_linear(x, weight, bias, mask_scale=0.05).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_masked_qk_matches_plain_qk(self) -> None:
        from tee_gpu_demo.masked_ops import masked_qk

        q = torch.randn(2, 4, 8)
        k = torch.randn(2, 6, 8)

        expected = q @ k.transpose(-1, -2)
        actual = masked_qk(q, k, rank=3, mask_scale=0.05).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_masked_pv_matches_plain_pv(self) -> None:
        from tee_gpu_demo.masked_ops import masked_pv

        p = torch.softmax(torch.randn(2, 4, 6), dim=-1)
        v = torch.randn(2, 6, 8)

        expected = p @ v
        actual = masked_pv(p, v, rank_p=3, rank_v=2, mask_scale=0.05).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_masked_kv_cache_matches_plain_cache(self) -> None:
        from tee_gpu_demo.masked_ops import MaskedKVCache

        cache = MaskedKVCache(dim=8, key_rank=3, query_rank=2, device=torch.device("cpu"))
        cache.append(torch.randn(4, 8))
        cache.append(torch.randn(3, 8))

        q = torch.randn(2, 8)
        expected = cache.baseline_query(q)
        actual = cache.query(q).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_masked_attention_cache_matches_plain_attention(self) -> None:
        from tee_gpu_demo.masked_ops import MaskedAttentionCache

        cache = MaskedAttentionCache(
            dim=8,
            key_rank=3,
            query_rank=2,
            prob_rank=3,
            value_rank=2,
            device=torch.device("cpu"),
        )
        cache.append(torch.randn(4, 8), torch.randn(4, 8))
        cache.append(torch.randn(3, 8), torch.randn(3, 8))

        q = torch.randn(2, 8)
        expected = cache.baseline_query(q)
        actual = cache.query(q).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_masked_attention_cache_applies_attention_mask(self) -> None:
        from tee_gpu_demo.masked_ops import MaskedAttentionCache

        cache = MaskedAttentionCache(
            dim=8,
            key_rank=3,
            query_rank=2,
            prob_rank=3,
            value_rank=2,
            device=torch.device("cpu"),
        )
        keys = torch.randn(5, 8)
        values = torch.randn(5, 8)
        cache.append(keys, values)

        q = torch.randn(3, 8)
        attention_mask = torch.zeros(3, 5)
        attention_mask[0, 3:] = torch.finfo(torch.float32).min
        attention_mask[1, 4:] = torch.finfo(torch.float32).min

        scores = q @ keys.transpose(-1, -2) / (8**0.5)
        expected = torch.softmax(scores + attention_mask, dim=-1) @ values
        actual = cache.query(q, attention_mask=attention_mask).output

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_batched_masked_attention_query_matches_plain_attention(self) -> None:
        from tee_gpu_demo.masked_ops import MaskedAttentionCache, batched_masked_attention_query

        caches = []
        queries = []
        for index, tokens in enumerate((4, 6, 5)):
            cache = MaskedAttentionCache(
                dim=8,
                key_rank=3,
                query_rank=2,
                prob_rank=3,
                value_rank=2,
                device=torch.device("cpu"),
            )
            cache.append(torch.randn(tokens, 8), torch.randn(tokens, 8))
            caches.append(cache)
            queries.append(torch.randn(1 + (index % 2), 8))

        actual = batched_masked_attention_query(caches, queries)
        for cache, query, result in zip(caches, queries, actual):
            expected = cache.baseline_query(query)
            self.assertTrue(torch.allclose(result.output, expected, atol=1e-5, rtol=1e-5))

    def test_llama_attention_patch_uses_masked_attention_cache(self) -> None:
        from tee_gpu_demo.llama_patch import replace_llama_attentions

        class FakeAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(num_attention_heads=2, num_key_value_heads=2, hidden_size=8)
                self.layer_idx = 0
                self.head_dim = 4
                self.num_key_value_groups = 1
                self.attention_dropout = 0.0
                self.scaling = self.head_dim**-0.5
                self.q_proj = nn.Linear(8, 8, bias=False)
                self.k_proj = nn.Linear(8, 8, bias=False)
                self.v_proj = nn.Linear(8, 8, bias=False)
                self.o_proj = nn.Linear(8, 8, bias=False)

            def forward(self, hidden_states, **kwargs):
                raise NotImplementedError

        model = nn.Module()
        model.attn = FakeAttention()
        hidden_states = torch.randn(1, 4, 8)
        attention_mask = torch.zeros(1, 1, 4, 4)
        attention_mask = attention_mask.masked_fill(
            torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1).view(1, 1, 4, 4),
            torch.finfo(torch.float32).min,
        )

        q = model.attn.q_proj(hidden_states).view(1, 4, 2, 4).transpose(1, 2)
        k = model.attn.k_proj(hidden_states).view(1, 4, 2, 4).transpose(1, 2)
        v = model.attn.v_proj(hidden_states).view(1, 4, 2, 4).transpose(1, 2)
        probs = torch.softmax((q @ k.transpose(-1, -2)) * model.attn.scaling + attention_mask, dim=-1)
        expected = probs @ v
        expected = expected.transpose(1, 2).contiguous().reshape(1, 4, 8)
        expected = model.attn.o_proj(expected)

        report = replace_llama_attentions(model, trusted_device="cpu", untrusted_device="cpu")
        self.assertEqual(report.replaced, 1)
        cos = torch.ones(1, 4, 4)
        sin = torch.zeros(1, 4, 4)
        actual, weights = model.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            output_attentions=True,
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(tuple(weights.shape), (1, 2, 4, 4))

    def test_llama_attention_patch_shares_masked_gqa_kv_cache(self) -> None:
        from tee_gpu_demo.llama_patch import replace_llama_attentions
        from tee_gpu_demo.masked_ops import MaskedAttentionCache

        class FakeGqaAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=8)
                self.layer_idx = 0
                self.head_dim = 2
                self.num_key_value_groups = 2
                self.attention_dropout = 0.0
                self.scaling = self.head_dim**-0.5
                self.q_proj = nn.Linear(8, 8, bias=False)
                self.k_proj = nn.Linear(8, 4, bias=False)
                self.v_proj = nn.Linear(8, 4, bias=False)
                self.o_proj = nn.Linear(8, 8, bias=False)

            def forward(self, hidden_states, **kwargs):
                raise NotImplementedError

        def repeat_kv(hidden_states, n_rep):
            batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch,
                num_key_value_heads,
                n_rep,
                seq_len,
                head_dim,
            )
            return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)

        model = nn.Module()
        model.attn = FakeGqaAttention()
        hidden_states = torch.randn(1, 4, 8)
        attention_mask = torch.zeros(1, 1, 4, 4)
        attention_mask = attention_mask.masked_fill(
            torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1).view(1, 1, 4, 4),
            torch.finfo(torch.float32).min,
        )

        q = model.attn.q_proj(hidden_states).view(1, 4, 4, 2).transpose(1, 2)
        k = model.attn.k_proj(hidden_states).view(1, 4, 2, 2).transpose(1, 2)
        v = model.attn.v_proj(hidden_states).view(1, 4, 2, 2).transpose(1, 2)
        k = repeat_kv(k, model.attn.num_key_value_groups)
        v = repeat_kv(v, model.attn.num_key_value_groups)
        probs = torch.softmax((q @ k.transpose(-1, -2)) * model.attn.scaling + attention_mask, dim=-1)
        expected = probs @ v
        expected = expected.transpose(1, 2).contiguous().reshape(1, 4, 8)
        expected = model.attn.o_proj(expected)

        report = replace_llama_attentions(model, trusted_device="cpu", untrusted_device="cpu")
        self.assertEqual(report.replaced, 1)
        cos = torch.ones(1, 4, 2)
        sin = torch.zeros(1, 4, 2)
        query_shapes = []
        original_query = MaskedAttentionCache.query

        def counting_query(cache, query, *args, **kwargs):
            query_shapes.append(tuple(query.shape))
            return original_query(cache, query, *args, **kwargs)

        MaskedAttentionCache.query = counting_query
        try:
            actual, weights = model.attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                output_attentions=True,
                use_cache=True,
            )
        finally:
            MaskedAttentionCache.query = original_query

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(tuple(weights.shape), (1, 4, 4, 4))
        self.assertEqual(query_shapes, [(8, 2), (8, 2)])
        cache_row = model.attn._tee_gpu_attention_caches[0]
        self.assertEqual(len(cache_row), 2)
        self.assertTrue(all(cache.private_keys.shape[0] == 4 for cache in cache_row))

    def test_continuous_batching_matches_plain_attention(self) -> None:
        from tee_gpu_demo.continuous_batching import (
            ContinuousBatchingEngine,
            ContinuousRequest,
            plain_attention_reference,
        )

        requests = []
        for index in range(3):
            prompt_len = 3 + index
            decode_len = 2 + (index % 2)
            requests.append(
                ContinuousRequest(
                    request_id=f"req-{index}",
                    prompt_keys=torch.randn(prompt_len, 8),
                    prompt_values=torch.randn(prompt_len, 8),
                    decode_queries=torch.randn(decode_len, 8),
                    decode_keys=torch.randn(decode_len, 8),
                    decode_values=torch.randn(decode_len, 8),
                    arrival_step=index,
                )
            )

        engine = ContinuousBatchingEngine(
            dim=8,
            max_batch_size=2,
            prefill_chunk=2,
            key_rank=2,
            query_rank=2,
            prob_rank=2,
            value_rank=2,
            trusted_device="cpu",
            untrusted_device="cpu",
        )
        engine.submit_many(requests)
        result = engine.run()

        self.assertGreater(result.max_decode_batch, 1)
        self.assertEqual(set(result.outputs), {request.request_id for request in requests})
        for request in requests:
            expected = plain_attention_reference(request)
            actual = result.outputs[request.request_id]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
