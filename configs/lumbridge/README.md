# Lumbridge configs

| File | What |
|---|---|
| `spark-smoke.toml` | 5-step fake-data SFT. Run by `scripts/lumbridge/spark-setup.sh` after install. |
| `bolt-e4b-smoke.toml` | 5-step fake-data LoRA SFT of Nemotron 3.5 Bolt-E4B through the Gemma-4 adapter. |

Real training jobs live in [lumbridge-train](https://github.com/lumbridge-corp/lumbridge-train), which writes an SFT config per run and calls `uv run sft` here.
