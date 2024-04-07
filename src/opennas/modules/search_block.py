from typing import Optional, Iterable, Any

import numpy as np
from torch import nn


class SearchBlock(nn.Module):

    def __init__(self, op_choices: Optional[Iterable[nn.Module]] = None):
        super().__init__()
        self.op_choices = nn.ModuleList[op_choices]
        self.forward_op_idx: Optional[int] = None

    def select_forward_op(self, idx: Optional[int]) -> None:
        if not (idx is None or 0 <= idx < len(self.op_choices)):
            raise ValueError(
                f'Operation index should be an integer from '
                f'0 up to {len(self.op_choices) - 1}, but got: {idx}'
            )
        self.forward_op_idx = idx

    def sample_random_op(self) -> Optional[int]:
        if not self.op_choices:
            return None
        idx = np.random.randint(0, len(self.op_choices))
        self.forward_op_idx = idx
        return idx

    def forward(self, *args, **kwargs) -> Any:
        if self.forward_op_idx is None:
            raise RuntimeError(
                'Forward index in search block is not specified'
            )
        return self.op_choices[self.forward_op_idx](*args, **kwargs)
