"""Checks the lumbridge-rl Gemma 4 adapter against stock transformers on real weights.

    uv run --no-sync python scripts/lumbridge/check_gemma4.py [model] [image.png]

1. one document: the packed path matches the stock HF forward (its own masks and positions).
2. packing: a document's hidden states are the same alone and packed after another document.
3. image (optional): a yes/no question about the image still gets a confident, right answer.
Needs about 17 GB of GPU memory for Bolt-E4B in bf16.
"""

import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoProcessor

import prime_rl.trainer.models  # noqa: F401  registers the PrimeRL classes
from prime_rl.trainer.models.gemma4 import Gemma4ForConditionalGeneration
from prime_rl.trainer.models.gemma4.modeling_gemma4 import Gemma4Model

MODEL = sys.argv[1] if len(sys.argv) > 1 else "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else None
TOL = 0.05  # bf16, 42 layers: compare relative error of the final hidden states


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def logits(model, hidden: torch.Tensor) -> torch.Tensor:
    cap = model.lm_head.softcap
    out = model.lm_head(hidden).float()
    return torch.tanh(out / cap) * cap


@torch.no_grad()
def main() -> None:
    config = AutoConfig.from_pretrained(MODEL)
    model = Gemma4ForConditionalGeneration.from_pretrained(MODEL, config=config, dtype=torch.bfloat16).cuda().eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight, "lm_head is not tied"
    print(f"loaded {type(model).__name__}, attn={model.config.get_text_config()._attn_implementation}")

    a = tok("The capital of France is Paris. The capital of Japan is", return_tensors="pt").input_ids.cuda()
    # Longer than the 512-token sliding window, so the sliding-mask path is exercised too.
    b = tok(
        " ".join(["Approve the deploy to production only if the tests pass."] * 60), return_tensors="pt"
    ).input_ids.cuda()
    print(f"doc a {a.shape[1]} tokens, doc b {b.shape[1]} tokens")

    ref_b = Gemma4Model.forward(model.model, input_ids=b, use_cache=False).last_hidden_state
    ours_b = model.model(input_ids=b, seq_lens=torch.tensor([b.shape[1]])).last_hidden_state
    e1 = rel_err(ours_b, ref_b)
    agree = (logits(model, ours_b[:, -1]).argmax() == logits(model, ref_b[:, -1]).argmax()).item()
    print(f"1. single doc vs stock HF: rel err {e1:.4f}, next-token agree {agree}")

    packed = torch.cat([a, b], dim=1)
    ours_packed = model.model(input_ids=packed, seq_lens=torch.tensor([a.shape[1], b.shape[1]])).last_hidden_state
    e2 = rel_err(ours_packed[:, a.shape[1] :], ours_b)
    leak = model.model(input_ids=packed, seq_lens=torch.tensor([packed.shape[1]])).last_hidden_state
    e2_leak = rel_err(leak[:, a.shape[1] :], ours_b)
    print(f"2. packed vs alone: rel err {e2:.4f} (one unmasked doc for contrast: {e2_leak:.4f})")

    ok = e1 < TOL and agree and e2 < TOL and e2_leak > 2 * e2
    if IMAGE:
        from PIL import Image

        image = Image.open(IMAGE).convert("RGB")
        yes, no = tok.encode("yes", add_special_tokens=False)[0], tok.encode("no", add_special_tokens=False)[0]
        for question, want in [("Is there a Post button on this screen?", True), ("Is this a bank login page?", False)]:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": f"{question} Answer yes or no."}],
                }
            ]
            prompt = proc.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
            inputs = proc(text=prompt, images=image, return_tensors="pt").to("cuda")
            ids = inputs.pop("input_ids")
            inputs.pop("attention_mask", None)
            mm = {k: v for k, v in inputs.items() if k in ("pixel_values", "image_position_ids", "mm_token_type_ids")}
            hidden = model.model(input_ids=ids, seq_lens=torch.tensor([ids.shape[1]]), **mm).last_hidden_state
            p = F.softmax(logits(model, hidden[:, -1])[0, [yes, no]], dim=-1)[0].item()
            right = (p > 0.5) == want
            ok = ok and right
            print(f"3. image: {question!r} p_yes={p:.3f} {'ok' if right else 'WRONG'}")

    print(f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
