#!/usr/bin/env bash
# Source this file to put DFTB+ + SK paths on PATH for offline pins.
# Installs live outside the git tree (see software/README.md next to molcrafts).
#
#   source tools/tb_oracle/setup_dftbplus_env.sh          # 24.1 (default)
#   ATV_DFTBPLUS_HOST=25.1 source tools/tb_oracle/...     # 25.1 development
#   dftb+ --version
#
# Two hosts are pinned because two are in use:
#
#   24.1   molcrafts/software/dftbplus-24.1  + software/dftb-params/mio-1-1
#   25.1   ../.conda/envs/dftbplus           + ../.conda/envs/dftbplus-sk/mio-main
#          (development build, commit a23bfb2)
#
# The two mio trees are **byte-identical** for every H/C/N/O pair — verified by
# diff, not assumed — so a pin taken on one host is valid on the other, and the
# H2O / C-chain pins reproduce to <1e-7 Ha across them. Override with
# ATV_DFTBPLUS_ROOT / ATV_DFTB_PARAMS_ROOT to install elsewhere.

: "${ATV_DFTBPLUS_HOST:=24.1}"
_atv_conda_envs=/nobackup/proj/disk/teoroo/personal/jicli594/work/.conda/envs

if [[ "${ATV_DFTBPLUS_HOST}" == "25.1" ]]; then
  : "${ATV_DFTBPLUS_ROOT:=${_atv_conda_envs}/dftbplus}"
  : "${ATV_DFTB_PARAMS_ROOT:=${_atv_conda_envs}/dftbplus-sk}"
  : "${ATV_DFTBPLUS_SK_MIO_DIR:=mio-main/skfiles}"
else
  : "${ATV_DFTBPLUS_ROOT:=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/software/dftbplus-24.1}"
  : "${ATV_DFTB_PARAMS_ROOT:=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/software/dftb-params}"
  : "${ATV_DFTBPLUS_SK_MIO_DIR:=mio-1-1}"
fi

if [[ ! -x "${ATV_DFTBPLUS_ROOT}/bin/dftb+" ]]; then
  echo "error: dftb+ not found at ${ATV_DFTBPLUS_ROOT}/bin/dftb+" >&2
  echo "  expected prebuilt 24.1 under molcrafts/software/ (see software/README.md)" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${ATV_DFTBPLUS_ROOT}/bin:${PATH}"
export ATV_DFTBPLUS_SK_MIO="${ATV_DFTB_PARAMS_ROOT}/${ATV_DFTBPLUS_SK_MIO_DIR}"
export ATV_DFTBPLUS_SK_3OB="${ATV_DFTB_PARAMS_ROOT}/3ob-3-1"

if [[ ! -f "${ATV_DFTBPLUS_SK_MIO}/H-H.skf" ]]; then
  echo "warning: mio SK missing at ${ATV_DFTBPLUS_SK_MIO}/H-H.skf" >&2
fi

echo "DFTB+ ready: $(command -v dftb+)"
dftb+ --version 2>&1 | head -3
echo "  ATV_DFTBPLUS_SK_MIO=${ATV_DFTBPLUS_SK_MIO}"
echo "  ATV_DFTBPLUS_SK_3OB=${ATV_DFTBPLUS_SK_3OB}"
