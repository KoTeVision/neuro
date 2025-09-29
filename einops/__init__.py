"""Lightweight fallback for the `einops` package used by MONAI.

This module only implements `rearrange` for pure dimension permutation
patterns (e.g. "b c d h w -> b d h w c") which is sufficient for the
SwinUNETR network configuration used in this project.

If the real `einops` package is available it will take precedence, but
this stub allows inference to run in environments where installing the
optional dependency is not possible.
"""
from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch always available in this app
    torch = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore


class RearrangeError(ValueError):
    """Raised when the pattern cannot be satisfied by this lightweight stub."""


def _parse_pattern(pattern: str) -> tuple[list[str], list[str]]:
    if "->" not in pattern:
        raise RearrangeError("Pattern must contain '->' separating input and output axes")
    left, right = (part.strip() for part in pattern.split("->", 1))
    if not left or not right:
        raise RearrangeError("Pattern must specify both input and output axes")
    left_axes = [token for token in left.split() if token]
    right_axes = [token for token in right.split() if token]
    if not left_axes or not right_axes:
        raise RearrangeError("Pattern must list axes on both sides")
    return left_axes, right_axes


def rearrange(tensor, pattern: str, *_, **__):
    """Perform axis permutation according to a simple einops-style pattern.

    Only pure permutations are supported; reshaping, axis creation, or reduction
    operations are not implemented in this fallback.
    """
    left_axes, right_axes = _parse_pattern(pattern)

    if not hasattr(tensor, "ndim"):
        raise RearrangeError(f"Object does not expose ndim: {type(tensor)!r}")

    if len(left_axes) != tensor.ndim:
        raise RearrangeError(
            f"Pattern expects {len(left_axes)} dimensions, received tensor with shape {getattr(tensor, 'shape', '?')}"
        )

    axis_order = {name: idx for idx, name in enumerate(left_axes)}

    try:
        perm = [axis_order[name] for name in right_axes]
    except KeyError as exc:  # pragma: no cover - defensive path
        raise RearrangeError(f"Axis '{exc.args[0]}' not present in pattern '{pattern}'") from exc

    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.permute(*perm)
    if _np is not None and isinstance(tensor, _np.ndarray):
        return tensor.transpose(perm)

    raise RearrangeError(f"Unsupported tensor type: {type(tensor)!r}")


__all__ = ["rearrange", "RearrangeError"]
