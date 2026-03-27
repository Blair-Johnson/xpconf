"""
xpconf — a minimal ML config library.

Features:
  - Dot access with optional auto-nesting of sub-dicts
  - Callable-as-reference: zero-arg callables are resolved on read
  - Freeze support: lock the config to prevent mutation
  - Lossless save/load via cloudpickle (preserves callables)
  - Lossy YAML export for human-readable snapshots
"""

from __future__ import annotations

import copy
import logging
import pickle
import yaml
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import cloudpickle as _pickle
except ImportError:
    _pickle = pickle  # type: ignore[assignment]


class ConfigDict:
    """
    A dot-accessible configuration dictionary.

    Zero-arg callables stored as values are resolved on access, enabling
    reactive references and derived values:

        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.mlp_dim = lambda: cfg.hidden_dim * 4
        cfg.mlp_dim  # → 3072

        cfg.hidden_dim = 1024
        cfg.mlp_dim  # → 4096

    Args:
        auto_nest: If True, accessing a missing attribute auto-creates a
            nested ConfigDict (which inherits this flag). If False, raises
            AttributeError on missing keys.
    """

    # Attributes that live on the object itself, not in the data store.
    _RESERVED = frozenset({"_data", "_frozen", "_auto_nest"})

    def __init__(self, auto_nest: bool = True, **kwargs: Any):
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "_auto_nest", auto_nest)
        for k, v in kwargs.items():
            self[k] = v

    # ── Resolution ─────────────────────────────────────────

    @staticmethod
    def _resolve(value: Any) -> Any:
        """If value is a zero-arg callable (not a type), call it."""
        if callable(value) and not isinstance(value, type):
            try:
                # Inspect isn't reliable on all callables, so we just try
                # calling with no args and catch TypeError.
                return value()
            except TypeError:
                return value
        return value

    @staticmethod
    def _is_callable_ref(value: Any) -> bool:
        """Check if a value is a callable reference (not a type/class)."""
        return callable(value) and not isinstance(value, type)

    # ── Access ─────────────────────────────────────────────

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        data = object.__getattribute__(self, "_data")
        if key in data:
            return self._resolve(data[key])
        if object.__getattribute__(self, "_frozen"):
            raise KeyError(key)
        if object.__getattribute__(self, "_auto_nest"):
            child = ConfigDict(auto_nest=True)
            data[key] = child
            return child
        raise AttributeError(
            f"ConfigDict has no attribute {key!r} (auto_nest is disabled)"
        )

    def __setattr__(self, key: str, value: Any):
        if key in self._RESERVED:
            object.__setattr__(self, key, value)
            return
        self._check_frozen()
        data = object.__getattribute__(self, "_data")
        data[key] = value

    def __delattr__(self, key: str):
        self._check_frozen()
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"ConfigDict has no attribute {key!r}")
        del data[key]

    # ── Dict-style access ──────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any):
        self.__setattr__(key, value)

    def __delitem__(self, key: str):
        self.__delattr__(key)

    def __contains__(self, key: str) -> bool:
        data = object.__getattribute__(self, "_data")
        return key in data

    def __iter__(self) -> Iterator[str]:
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def __len__(self) -> int:
        data = object.__getattribute__(self, "_data")
        return len(data)

    # ── Freezing ───────────────────────────────────────────

    def freeze(self) -> ConfigDict:
        """Freeze this dict and all nested ConfigDicts. Returns self."""
        object.__setattr__(self, "_frozen", True)
        data = object.__getattribute__(self, "_data")
        for v in data.values():
            if isinstance(v, ConfigDict):
                v.freeze()
        return self

    def unfreeze(self) -> ConfigDict:
        """Unfreeze this dict and all nested ConfigDicts. Returns self."""
        object.__setattr__(self, "_frozen", False)
        data = object.__getattribute__(self, "_data")
        for v in data.values():
            if isinstance(v, ConfigDict):
                v.unfreeze()
        return self

    @property
    def is_frozen(self) -> bool:
        return object.__getattribute__(self, "_frozen")

    def _check_frozen(self):
        if object.__getattribute__(self, "_frozen"):
            raise FrozenError("Cannot mutate a frozen ConfigDict")

    # ── Raw access (bypass resolution) ─────────────────────

    def get_raw(self, key: str) -> Any:
        """Get the raw stored value without resolving callables."""
        data = object.__getattribute__(self, "_data")
        if key not in data:
            raise AttributeError(f"ConfigDict has no attribute {key!r}")
        return data[key]

    # ── Dotpath access ─────────────────────────────────────

    def get_by_dotpath(self, path: str) -> Any:
        """
        Get a value by dotpath string.

            cfg.get_by_dotpath('model.backbone.hidden_dim')
            # equivalent to cfg.model.backbone.hidden_dim
        """
        obj = self
        parts = path.split(".")
        for i, part in enumerate(parts):
            try:
                obj = obj[part]
            except TypeError:
                traversed = ".".join(parts[:i])
                raise KeyError(
                    f"Cannot traverse into {traversed!r} "
                    f"(got {type(obj).__name__}, expected ConfigDict)"
                ) from None
        return obj

    def set_by_dotpath(self, path: str, value: Any, *, coerce: bool = False):
        """
        Set a value by dotpath string.

            cfg.set_by_dotpath('model.backbone.hidden_dim', 1024)
            # equivalent to cfg.model.backbone.hidden_dim = 1024

        If coerce=True and value is a string, it will be cast to match
        the type of the existing value (supports bool, int, float, str).
        """
        parts = path.split(".")
        obj = self
        for part in parts[:-1]:
            obj = obj[part]
        if coerce and isinstance(value, str):
            value = self._coerce_value(value, obj[parts[-1]])
        obj[parts[-1]] = value

    @staticmethod
    def _coerce_value(value: str, existing: Any) -> Any:
        """Cast a string value to match the type of an existing value."""
        if isinstance(existing, bool):
            return value.lower() in ("true", "1", "yes")
        if isinstance(existing, int):
            return int(value)
        if isinstance(existing, float):
            return float(value)
        return value

    # ── Serialization (lossless — pickle) ────────────────────

    def save(self, path: str):
        """
        Save the config losslessly via cloudpickle/pickle.
        Preserves callables, references, and the full object graph.
        Ideal for checkpointing alongside model weights.
        """
        with open(path, "wb") as f:
            _pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> ConfigDict:
        """
        Load a config saved with save(). Fully restores callables
        and references — mutations to the loaded config propagate
        through its lambdas as expected.
        """
        with open(path, "rb") as f:
            cfg = pickle.load(f)
        if not isinstance(cfg, ConfigDict):
            raise TypeError(
                f"Expected a ConfigDict, got {type(cfg).__name__}. "
                f"Use ConfigDict.load_yaml() for YAML files."
            )
        return cfg

    # ── Serialization (lossy — YAML) ──────────────────────

    def to_dict(self) -> dict:
        """
        Resolve all values (including callables) and return a plain dict.
        Nested ConfigDicts become nested dicts.
        """
        return self._to_dict_inner()

    def _to_dict_inner(
        self, _prefix: str = "", _callable_paths: list | None = None,
    ) -> dict:
        data = object.__getattribute__(self, "_data")
        out = {}
        for k, v in data.items():
            path = k if not _prefix else f"{_prefix}.{k}"
            if self._is_callable_ref(v) and _callable_paths is not None:
                _callable_paths.append(path)
            resolved = self._resolve(v)
            if isinstance(resolved, ConfigDict):
                resolved = resolved._to_dict_inner(
                    _prefix=path, _callable_paths=_callable_paths,
                )
            out[k] = resolved
        return out

    def _to_dict_with_warnings(self) -> dict:
        """to_dict that logs warnings for callable fields being resolved."""
        callable_paths: list[str] = []
        result = self._to_dict_inner(_callable_paths=callable_paths)
        if callable_paths:
            paths_str = ", ".join(callable_paths)
            logger.warning(
                "YAML serialization resolved %d callable(s) to their current "
                "values (lossy): [%s]. Use save() for lossless serialization.",
                len(callable_paths),
                paths_str,
            )
        return result

    def to_yaml(self) -> str:
        """Serialize to YAML string. Lossy: callables resolve to current values."""
        return yaml.dump(
            self._to_dict_with_warnings(),
            default_flow_style=False,
            sort_keys=False,
        )

    def save_yaml(self, path: str):
        """
        Save a human-readable YAML snapshot. Lossy: callables are
        resolved to their current values. Use save() for lossless
        serialization.
        """
        with open(path, "w") as f:
            yaml.dump(
                self._to_dict_with_warnings(),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_dict(cls, d: dict, auto_nest: bool = True) -> ConfigDict:
        """Create a ConfigDict from a plain dict. Nested dicts become ConfigDicts."""
        cfg = cls(auto_nest=auto_nest)
        for k, v in d.items():
            if isinstance(v, dict):
                v = cls.from_dict(v, auto_nest=auto_nest)
            cfg[k] = v
        return cfg

    @classmethod
    def load_yaml(cls, path: str, auto_nest: bool = True) -> ConfigDict:
        """
        Load a ConfigDict from a YAML file. Plain values only —
        no callables or references. For simple configs that don't
        need the Python config-as-code pattern.
        """
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        if d is None:
            return cls(auto_nest=auto_nest)
        return cls.from_dict(d, auto_nest=auto_nest)

    # ── Representation ─────────────────────────────────────

    def __repr__(self) -> str:
        frozen_tag = " [FROZEN]" if self.is_frozen else ""
        return f"ConfigDict({self._repr_inner()}){frozen_tag}"

    def _repr_inner(self, indent: int = 1) -> str:
        data = object.__getattribute__(self, "_data")
        if not data:
            return "{}"
        pad = "  " * indent
        pad_close = "  " * (indent - 1)
        lines = []
        for k, v in data.items():
            if isinstance(v, ConfigDict):
                inner = v._repr_inner(indent + 1)
                lines.append(f"{pad}{k}: {inner}")
            elif self._is_callable_ref(v):
                try:
                    resolved = v()
                    lines.append(f"{pad}{k}: {resolved!r}  # <callable>")
                except Exception:
                    lines.append(f"{pad}{k}: <callable>")
            else:
                lines.append(f"{pad}{k}: {v!r}")
        return "{\n" + "\n".join(lines) + f"\n{pad_close}}}"

    def keys(self):
        data = object.__getattribute__(self, "_data")
        return data.keys()

    def values(self):
        data = object.__getattribute__(self, "_data")
        return [self._resolve(v) for v in data.values()]

    def items(self):
        data = object.__getattribute__(self, "_data")
        return [(k, self._resolve(v)) for k, v in data.items()]

    def update(self, other: dict | ConfigDict, **kwargs):
        """Update from a dict, ConfigDict, or keyword args."""
        self._check_frozen()
        if isinstance(other, ConfigDict):
            other_data = object.__getattribute__(other, "_data")
            for k, v in other_data.items():
                self[k] = v
        elif isinstance(other, dict):
            for k, v in other.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v


class FrozenError(Exception):
    """Raised when attempting to mutate a frozen ConfigDict."""
    pass
