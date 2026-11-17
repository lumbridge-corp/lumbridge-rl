#!/usr/bin/env bash
# Install lumbridge-rl on an NVIDIA DGX Spark (GB10, aarch64, CUDA 13) and run a 5-step SFT smoke test.
# Low priority and capped build jobs, so it can run next to models that are already serving.
#
#   scripts/lumbridge/spark-setup.sh            # install + smoke
#   SKIP_SMOKE=1 scripts/lumbridge/spark-setup.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
[ -x "$UV" ] || { echo "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

# Keep the build from starving the box: few compile jobs, idle I/O class.
export MAX_JOBS=${MAX_JOBS:-2} NVCC_THREADS=1 CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-2} UV_CONCURRENT_BUILDS=1
# mamba-ssm / causal-conv1d CUDA builds are only needed for Mamba-hybrid trainer models.
export MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=FALSE

echo "== submodules"
git submodule update --init --filter=blob:none --jobs 1 || exit 1

echo "== uv sync $(date -Is)"
# flashinfer's prebuilt JIT cache and cubins have no aarch64 + sm_12x build; vLLM falls back without them.
nice -n 19 ionice -c3 "$UV" sync --extra gpu --extra flash-attn \
  --no-install-package flashinfer-jit-cache --no-install-package flashinfer-cubin || exit 1

echo "== check"
"$UV" run --no-sync python -c "import flash_attn, torch; print('flash-attn', flash_attn.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '|', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" || exit 1

[ -n "${SKIP_SMOKE:-}" ] && exit 0
echo "== smoke sft $(date -Is)"
nice -n 19 "$UV" run --no-sync sft @ configs/lumbridge/spark-smoke.toml --no-monitors.prime
