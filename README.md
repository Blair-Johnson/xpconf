# xpconf

A minimal ML configuration library. One file, no magic.

## Features

- **Dot access** with optional auto-nesting of sub-dicts
- **Callable references** — zero-arg callables (lambdas) are resolved on read, giving you reactive derived values
- **Freeze support** — lock the config to prevent accidental mutation during training
- **Lossless save/load** — cloudpickle preserves callables and the full object graph
- **Lossy YAML export** — human-readable snapshots for logging, with warnings when callables are flattened

## Install

```bash
pip install git+https://github.com/blair-johnson/xpconf.git
```

## Quick start

```python
from xpconf import ConfigDict

cfg = ConfigDict()

# Dot access with auto-nesting
cfg.model.backbone.hidden_dim = 768
cfg.model.backbone.num_layers = 12

# Reactive references via lambdas
cfg.model.mlp_dim = lambda: cfg.model.backbone.hidden_dim * 4
cfg.model.mlp_dim  # → 3072

cfg.model.backbone.hidden_dim = 1024
cfg.model.mlp_dim  # → 4096 (auto-updated)
```

## Derived values

Any zero-arg callable stored as a value is called on access. This gives you readonly references, computed values, and conditional logic — all with plain Python:

```python
cfg = ConfigDict()
cfg.hidden_dim = 768
cfg.num_heads = 12

# Derived values
cfg.head_dim = lambda: cfg.hidden_dim // cfg.num_heads
cfg.mlp_dim = lambda: cfg.hidden_dim * 4

# Readonly cross-references
cfg.head.input_dim = lambda: cfg.hidden_dim

# Conditional logic
cfg.use_bias = True
cfg.bias_dim = lambda: cfg.hidden_dim if cfg.use_bias else 0
```

## Freezing

Lock the config after setup to catch accidental mutations:

```python
cfg = ConfigDict()
cfg.lr = 3e-4
cfg.freeze()

cfg.lr = 1e-5       # → raises FrozenError
cfg.new_key = 42     # → raises FrozenError

# Reads (including lambda resolution) still work
print(cfg.lr)        # → 0.0003

# Unfreeze if you need to (e.g. for fine-tuning overrides)
cfg.unfreeze()
cfg.lr = 1e-5        # works now
```

## Auto-nesting control

By default, accessing a missing key auto-creates a nested `ConfigDict`. Disable this for strict schemas:

```python
# Permissive (default) — great for quick iteration
cfg = ConfigDict(auto_nest=True)
cfg.model.backbone.dim = 768  # auto-creates intermediate dicts

# Strict — catches typos
cfg = ConfigDict(auto_nest=False)
cfg.model  # → raises AttributeError
```

Children inherit the parent's `auto_nest` setting:

```python
cfg = ConfigDict(auto_nest=True)
cfg.model = ConfigDict(auto_nest=False)  # strict subtree
cfg.model.dim = 768       # explicit set works
cfg.model.backbone        # → raises AttributeError
cfg.data.path = "/data"   # parent still auto-nests
```

## Serialization

### Lossless (pickle) — for checkpoints

```python
# Save alongside model weights
cfg.save("checkpoints/run_042/config.pkl")

# Restore — lambdas, frozen state, everything survives
cfg = ConfigDict.load("checkpoints/run_042/config.pkl")
cfg.unfreeze()
cfg.train.lr = 1e-5  # override for fine-tuning
cfg.freeze()
```

### Lossy (YAML) — for logging and inspection

```python
# Human-readable snapshot
cfg.save_yaml("runs/run_042/config.yaml")

# Warns if callables are being flattened:
#   WARNING:xpconf:YAML serialization resolved 2 callable(s) to their
#   current values (lossy): [model.mlp_dim, head.input_dim].
#   Use save() for lossless serialization.

# Load simple YAML configs (no callables)
cfg = ConfigDict.load_yaml("configs/base.yaml")
```

## Composition pattern

Configs are just Python functions. Compose them however you want:

```python
from xpconf import ConfigDict

def vit_base():
    cfg = ConfigDict()
    cfg.num_layers = 12
    cfg.hidden_dim = 768
    cfg.num_heads = 12
    return cfg

def classification(backbone_fn, num_classes=1000):
    cfg = ConfigDict()
    cfg.backbone = backbone_fn()
    cfg.head.num_classes = num_classes
    cfg.head.input_dim = lambda: cfg.backbone.hidden_dim
    cfg.loss.name = "cross_entropy"
    return cfg

# Compose
cfg = classification(vit_base, num_classes=100)

# Ad-hoc ablation
cfg.backbone.num_layers = 6
```
