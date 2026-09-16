#!/usr/bin/env bash
# Run an offline DFTB+ pin case.
# Usage:
#   source tools/tb_oracle/setup_dftbplus_env.sh
#   tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/h2_mio_scc [mio|3ob]
set -euo pipefail
CASE_DIR=${1:?case directory with geo.gen and dftb_in.hsd}
SK_KIND=${2:-mio}
: "${ATV_DFTBPLUS_SK_MIO:?source setup_dftbplus_env.sh first}"
: "${ATV_DFTBPLUS_SK_3OB:?}"
if [[ "$SK_KIND" == "3ob" ]]; then
  SK="$ATV_DFTBPLUS_SK_3OB"
else
  SK="$ATV_DFTBPLUS_SK_MIO"
fi
WORKDIR=$(mktemp -d /tmp/atv-dftbplus-XXXX)
trap 'rm -rf "$WORKDIR"' EXIT
cp -a "$CASE_DIR/." "$WORKDIR/"
cd "$WORKDIR"
# Rewrite SlaterKoster Prefix to absolute SK dir if present
if [[ -f dftb_in.hsd ]]; then
  # inject absolute Prefix — cases often use Prefix = "./"
  sed -i "s|Prefix = \".*\"|Prefix = \"${SK}/\"|g" dftb_in.hsd
fi
dftb+
echo "---- results (workdir was $WORKDIR before trap) ----"
if [[ -f results.tag ]]; then
  grep -E 'total_energy|mermin_energy|forces' results.tag | head -20
fi
if [[ -f detailed.out ]]; then
  grep -E 'Total Energy|Force related' detailed.out | head -10
fi
