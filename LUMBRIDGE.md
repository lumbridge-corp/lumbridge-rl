# lumbridge-rl

A light fork of [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) (Apache-2.0),
modified by Lumbridge Corp to train small models on your own machine. Used by
[lumbridge-train](https://github.com/lumbridge-corp/lumbridge-train), the learning loop.

## Rules for this fork

- `upstream` = PrimeIntellect-ai/prime-rl. Merge upstream regularly.
- The Python package stays `prime_rl` so upstream merges stay cheap. Do not rename internals.
- Keep our changes small and in clearly named files where possible (`*lumbridge*`), and mark modified
  upstream files in the list below (Apache-2.0 section 4b).
- Keep `LICENSE` and upstream copyright notices.

## DGX Spark

`scripts/lumbridge/spark-setup.sh` installs this repo on an NVIDIA DGX Spark (GB10, aarch64, CUDA 13) and runs a
5-step SFT smoke test (`configs/lumbridge/spark-smoke.toml`). What it does differently from the upstream install:

- Skips flashinfer's prebuilt JIT cache and cubins (no aarch64 + sm_12x build); vLLM falls back without them.
- Skips the mamba-ssm and causal-conv1d CUDA builds (only Mamba-hybrid trainer models need them).
- Caps compile jobs and runs at idle priority, so it installs next to models that are already serving.

Flash-attention 2 is picked automatically on SM121. Single-GPU SFT of a 0.6B model runs at about 2k tokens/s.

## Mods

1. Done: DGX Spark profile (above).
2. Done: Gemma-4 model for Nemotron 3.5 Bolt-E4B (`src/prime_rl/trainer/models/gemma4/`). A thin adapter over
   transformers' own Gemma 4 modules: packed rows get explicit block-causal and sliding-window SDPA masks, weights load
   through DCP with no conversion, the head stays tied with Gemma's logit softcap. SDPA only (the global layers use
   head_dim 512, beyond flash-attention 2); the audio tower runs eager. No context parallelism yet.
   `scripts/lumbridge/check_gemma4.py` checks it against stock transformers (single doc, packing, an image question);
   `configs/lumbridge/bolt-e4b-smoke.toml` is a 5-step LoRA SFT.
3. Planned: decision loss: loss on the answer token(s) only, plus a temperature-calibration step.

## Modified upstream files

- `README.md`: a short note at the top that this is the Lumbridge fork.
- `.gitmodules`: HTTPS URLs instead of SSH, so a clone works without a GitHub SSH key.
- `src/prime_rl/trainer/models/__init__.py`: registers `gemma4` in the VLM mapping.
- `src/prime_rl/trainer/vlm.py`: `get_vision_encoder` also finds Gemma's `vision_tower`.
