#!/bin/bash

#Copyright (c) 2021, salesforce.com, inc.
#All rights reserved.
#SPDX-License-Identifier: BSD-3-Clause
#For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_17_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch
if V(torch.__version__) >= V("1.7"):
    print("true")
else:
    print("false")
EOF
)

torch_11_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch
if V(torch.__version__) >= V("1.1"):
    print("true")
else:
    print("false")
EOF
)

torch_version=$(python3 <<EOF
import torch
version = torch.__version__.split(".")
print(version[0] + version[1])
EOF
)

cuda_version=$(python3 <<EOF
import torch
if torch.cuda.is_available():
    version=torch.version.cuda.split(".")
    # 10.1.aa -> 101
    print(version[0] + version[1])
else:
    print("")
EOF
)
echo "cuda_version=${cuda_version}"

if "${torch_17_plus}"; then

    echo "[WARNING] warp-ctc is not prepared for pytorch>=1.7.0 now"

elif "${torch_11_plus}"; then

    warpctc_version=0.2.1
    if [ -z "${cuda_version}" ]; then
        python3 -m pip install warpctc-pytorch==${warpctc_version}+torch"${torch_version}".cpu
    else
        python3 -m pip install warpctc-pytorch==${warpctc_version}+torch"${torch_version}".cuda"${cuda_version}"
    fi

else
    echo "[WARNING] the asr package requires pytorch>=1.4.0 now"
fi