# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from transformers.activations import SiLUActivation

from .utils.runtime_utils import npu_is_cr13_or_later


class ExpSiLU(nn.Module):
    """SiLU written as `x / (1 + exp(-x))`.

    On RBLN-CR13 (rebel-compiler 0.12.1.dev163) `F.silu` and `x * sigmoid(x)` both lower to a fused SiLU kernel that
    loses precision: in the Qwen3-VL text MLPs, `silu(gate) * up` of layer 2 is off by 1.4 % (0.36 % for CPU bf16) and
    the error compounds over the layers. The exp form is lowered differently and stays at 0.1 %.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (1 + torch.exp(-x))


def replace_silu_for_npu(module: nn.Module, npu: str | None = None) -> nn.Module:
    """On RBLN-CR13 or later, replaces the SiLU activation modules in `module` with `ExpSiLU`, in place.

    Only `nn.SiLU` and the Hugging Face `SiLUActivation` modules are replaced; SiLU called as a function
    (`F.silu`) inside a forward is not. On other NPUs `module` is left unchanged.

    Args:
        module (nn.Module): The module to patch before tracing.
        npu (str | None): The target NPU, e.g. `rbln_config.npu`. Defaults to the attached device.

    Returns:
        nn.Module: `module` itself.
    """
    if not npu_is_cr13_or_later(npu):
        return module
    for parent in list(module.modules()):
        for name, child in parent.named_children():
            if isinstance(child, (nn.SiLU, SiLUActivation)):
                setattr(parent, name, ExpSiLU())
    return module
