#!/usr/bin/env bash
set -euo pipefail

# One-shot env build. quinine 0.3.0 is the only version on PyPI but it has
# stale strict pins (pyyaml==5.4 has no cp310 wheels and source-builds break
# under Cython >=3), so we install it last with --no-deps. Its runtime use of
# pyyaml/Cerberus/etc. is API-stable, so modern versions of those (installed
# by the env file above) work fine.
ENV_NAME=icl-metal

# Workaround for conda's signature-verification plugin failing on systems
# whose OpenSSL config doesn't expose the legacy provider.
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1

conda env create -f environment.yml
conda run -n "${ENV_NAME}" pip install --no-deps quinine
echo
echo "Env '${ENV_NAME}' ready. Activate with: conda activate ${ENV_NAME}"
