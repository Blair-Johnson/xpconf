"""Tests for xpconf."""

import os
import tempfile
import pytest
from xpconf import ConfigDict, FrozenError


# ── Basic dot access ──────────────────────────────────────

class TestBasicAccess:
    def test_set_and_get(self):
        cfg = ConfigDict()
        cfg.x = 10
        assert cfg.x == 10

    def test_dict_style_access(self):
        cfg = ConfigDict()
        cfg["x"] = 10
        assert cfg["x"] == 10
        assert cfg.x == 10

    def test_nested(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        assert cfg.model.dim == 768

    def test_contains(self):
        cfg = ConfigDict()
        cfg.x = 10
        assert "x" in cfg
        assert "y" not in cfg

    def test_delete(self):
        cfg = ConfigDict()
        cfg.x = 10
        del cfg.x
        assert "x" not in cfg

    def test_len(self):
        cfg = ConfigDict()
        cfg.a = 1
        cfg.b = 2
        assert len(cfg) == 2

    def test_iter(self):
        cfg = ConfigDict()
        cfg.a = 1
        cfg.b = 2
        assert list(cfg) == ["a", "b"]

    def test_keys_values_items(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = 20
        assert list(cfg.keys()) == ["x", "y"]
        assert list(cfg.values()) == [10, 20]
        assert list(cfg.items()) == [("x", 10), ("y", 20)]

    def test_update_from_dict(self):
        cfg = ConfigDict()
        cfg.x = 1
        cfg.update({"x": 2, "y": 3})
        assert cfg.x == 2
        assert cfg.y == 3

    def test_overwrite(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.x = 20
        assert cfg.x == 20


# ── Auto-nesting ──────────────────────────────────────────

class TestAutoNest:
    def test_auto_nest_creates_children(self):
        cfg = ConfigDict(auto_nest=True)
        cfg.model.backbone.dim = 768
        assert cfg.model.backbone.dim == 768

    def test_auto_nest_disabled_raises(self):
        cfg = ConfigDict(auto_nest=False)
        with pytest.raises(AttributeError, match="auto_nest is disabled"):
            _ = cfg.model

    def test_auto_nest_inherited(self):
        cfg = ConfigDict(auto_nest=True)
        cfg.model.backbone.dim = 768
        # The auto-created children should also auto-nest
        cfg.model.backbone.attention.num_heads = 12
        assert cfg.model.backbone.attention.num_heads == 12

    def test_auto_nest_disabled_inherited(self):
        cfg = ConfigDict(auto_nest=False)
        cfg.model = ConfigDict(auto_nest=False)
        cfg.model.dim = 768  # explicit set works
        with pytest.raises(AttributeError):
            _ = cfg.model.backbone  # but auto-nest doesn't

    def test_auto_nest_mixed(self):
        """Parent auto-nests, child doesn't."""
        cfg = ConfigDict(auto_nest=True)
        cfg.model = ConfigDict(auto_nest=False)
        cfg.model.dim = 768
        with pytest.raises(AttributeError):
            _ = cfg.model.backbone
        # But the parent still auto-nests
        cfg.data.path = "/data"
        assert cfg.data.path == "/data"


# ── Callable references ───────────────────────────────────

class TestCallableRefs:
    def test_lambda_resolved_on_access(self):
        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.mlp_dim = lambda: cfg.hidden_dim * 4
        assert cfg.mlp_dim == 3072

    def test_lambda_tracks_changes(self):
        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.mlp_dim = lambda: cfg.hidden_dim * 4
        assert cfg.mlp_dim == 3072

        cfg.hidden_dim = 1024
        assert cfg.mlp_dim == 4096

    def test_readonly_ref_pattern(self):
        """Lambda acts as a readonly reference — can't write through it."""
        cfg = ConfigDict()
        cfg.model.backbone.hidden_dim = 768
        cfg.head.input_dim = lambda: cfg.model.backbone.hidden_dim
        assert cfg.head.input_dim == 768

        cfg.model.backbone.hidden_dim = 1024
        assert cfg.head.input_dim == 1024

    def test_chained_lambdas(self):
        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.mlp_dim = lambda: cfg.hidden_dim * 4
        cfg.mlp_bias_size = lambda: cfg.mlp_dim  # chains through another lambda
        assert cfg.mlp_bias_size == 3072

        cfg.hidden_dim = 1024
        assert cfg.mlp_bias_size == 4096

    def test_lambda_with_conditionals(self):
        cfg = ConfigDict()
        cfg.use_bias = True
        cfg.hidden_dim = 768
        cfg.bias_dim = lambda: cfg.hidden_dim if cfg.use_bias else 0
        assert cfg.bias_dim == 768

        cfg.use_bias = False
        assert cfg.bias_dim == 0

    def test_classes_not_resolved(self):
        """Types/classes should be stored as-is, not called."""
        cfg = ConfigDict()
        cfg.activation = int  # a type, not a ref
        assert cfg.activation is int

    def test_callable_with_args_not_resolved(self):
        """Callables that need args should be stored as-is."""
        def needs_args(x, y):
            return x + y
        cfg = ConfigDict()
        cfg.fn = needs_args
        assert cfg.fn is needs_args

    def test_get_raw_returns_lambda(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        raw = cfg.get_raw("y")
        assert callable(raw)
        assert cfg.y == 20

    def test_values_resolves_callables(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        assert list(cfg.values()) == [10, 20]

    def test_items_resolves_callables(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        assert list(cfg.items()) == [("x", 10), ("y", 20)]


# ── Freezing ──────────────────────────────────────────────

class TestFreezing:
    def test_freeze_prevents_set(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        with pytest.raises(FrozenError):
            cfg.x = 20

    def test_freeze_prevents_new_keys(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        with pytest.raises(FrozenError):
            cfg.y = 20

    def test_freeze_prevents_delete(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        with pytest.raises(FrozenError):
            del cfg.x

    def test_freeze_is_recursive(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        cfg.freeze()
        with pytest.raises(FrozenError):
            cfg.model.dim = 1024

    def test_freeze_prevents_auto_nest(self):
        cfg = ConfigDict()
        cfg.model.dim = 768
        cfg.freeze()
        with pytest.raises(FrozenError):
            _ = cfg.typo

    def test_freeze_prevents_nested_auto_nest(self):
        cfg = ConfigDict()
        cfg.model.dim = 768
        cfg.freeze()
        with pytest.raises(FrozenError):
            _ = cfg.model.typo

    def test_unfreeze(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        cfg.unfreeze()
        cfg.x = 20
        assert cfg.x == 20

    def test_unfreeze_is_recursive(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        cfg.freeze()
        cfg.unfreeze()
        cfg.model.dim = 1024
        assert cfg.model.dim == 1024

    def test_is_frozen_property(self):
        cfg = ConfigDict()
        assert not cfg.is_frozen
        cfg.freeze()
        assert cfg.is_frozen

    def test_freeze_returns_self(self):
        cfg = ConfigDict()
        cfg.x = 10
        result = cfg.freeze()
        assert result is cfg

    def test_frozen_read_still_works(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        cfg.freeze()
        assert cfg.x == 10
        assert cfg.y == 20

    def test_freeze_prevents_update(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        with pytest.raises(FrozenError):
            cfg.update({"x": 20})


# ── Serialization ─────────────────────────────────────────

class TestSerialization:
    def test_to_dict_plain(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = "hello"
        assert cfg.to_dict() == {"x": 10, "y": "hello"}

    def test_to_dict_nested(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        assert cfg.to_dict() == {"model": {"dim": 768}}

    def test_to_dict_resolves_callables(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        d = cfg.to_dict()
        assert d == {"x": 10, "y": 20}

    # ── Lossless pickle save/load ─────────────────────────

    def test_pickle_roundtrip_plain(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        cfg.model.layers = 12
        cfg.lr = 3e-4

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            loaded = ConfigDict.load(path)
            assert loaded.model.dim == 768
            assert loaded.model.layers == 12
            assert loaded.lr == 3e-4
        finally:
            os.unlink(path)

    def test_pickle_preserves_callables(self):
        """Lambdas survive save/load and still react to mutations."""
        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.mlp_dim = lambda: cfg.hidden_dim * 4
        assert cfg.mlp_dim == 3072

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            loaded = ConfigDict.load(path)
            assert loaded.mlp_dim == 3072

            # Mutate the loaded config — lambda should follow
            loaded.hidden_dim = 1024
            assert loaded.mlp_dim == 4096

            # Original is unaffected
            assert cfg.hidden_dim == 768
            assert cfg.mlp_dim == 3072
        finally:
            os.unlink(path)

    def test_pickle_preserves_frozen_state(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            cfg.save(path)
            loaded = ConfigDict.load(path)
            assert loaded.is_frozen
            with pytest.raises(FrozenError):
                loaded.x = 20
        finally:
            os.unlink(path)

    def test_load_rejects_non_configdict(self):
        """load() should reject pickle files that aren't ConfigDicts."""
        import pickle as pkl
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pkl.dump({"x": 10}, f)
            path = f.name
        try:
            with pytest.raises(TypeError, match="Expected a ConfigDict"):
                ConfigDict.load(path)
        finally:
            os.unlink(path)

    # ── Lossy YAML save/load ──────────────────────────────

    def test_yaml_roundtrip(self):
        cfg = ConfigDict()
        cfg.model = ConfigDict()
        cfg.model.dim = 768
        cfg.model.layers = 12
        cfg.lr = 3e-4

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            cfg.save_yaml(path)
            loaded = ConfigDict.load_yaml(path)
            assert loaded.model.dim == 768
            assert loaded.model.layers == 12
            assert loaded.lr == 3e-4
        finally:
            os.unlink(path)

    def test_yaml_lossy_for_callables(self):
        """Callables resolve to values on YAML save — links are gone on load."""
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            cfg.save_yaml(path)
            loaded = ConfigDict.load_yaml(path)
            assert loaded.y == 20  # plain int, not a callable
            loaded.x = 999
            assert loaded.y == 20  # link is gone
        finally:
            os.unlink(path)

    def test_from_dict(self):
        d = {"model": {"dim": 768, "layers": 12}, "lr": 3e-4}
        cfg = ConfigDict.from_dict(d)
        assert cfg.model.dim == 768
        assert isinstance(cfg.model, ConfigDict)

    def test_from_dict_inherits_auto_nest(self):
        d = {"model": {"dim": 768}}
        cfg = ConfigDict.from_dict(d, auto_nest=False)
        assert cfg.model.dim == 768
        with pytest.raises(AttributeError):
            _ = cfg.model.backbone

    def test_to_yaml_string(self):
        cfg = ConfigDict()
        cfg.x = 10
        text = cfg.to_yaml()
        assert "x: 10" in text

    def test_load_yaml_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            cfg = ConfigDict.load_yaml(path)
            assert len(cfg) == 0
        finally:
            os.unlink(path)


# ── Repr ──────────────────────────────────────────────────

class TestRepr:
    def test_repr_basic(self):
        cfg = ConfigDict()
        cfg.x = 10
        r = repr(cfg)
        assert "x: 10" in r

    def test_repr_frozen(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        r = repr(cfg)
        assert "FROZEN" in r

    def test_repr_callable(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        r = repr(cfg)
        assert "callable" in r
        assert "20" in r

    def test_repr_empty(self):
        cfg = ConfigDict()
        r = repr(cfg)
        assert "ConfigDict({})" == r


# ── Constructor kwargs ────────────────────────────────────

class TestConstructorKwargs:
    def test_kwargs(self):
        cfg = ConfigDict(x=10, y=20)
        assert cfg.x == 10
        assert cfg.y == 20


# ── Edge cases ────────────────────────────────────────────

class TestEdgeCases:
    def test_none_values(self):
        cfg = ConfigDict()
        cfg.x = None
        assert cfg.x is None

    def test_list_values(self):
        cfg = ConfigDict()
        cfg.layers = [1, 2, 3]
        assert cfg.layers == [1, 2, 3]

    def test_bool_values(self):
        cfg = ConfigDict()
        cfg.flag = True
        assert cfg.flag is True

    def test_lambda_returning_none(self):
        cfg = ConfigDict()
        cfg.x = lambda: None
        assert cfg.x is None

    def test_overwrite_lambda_with_value(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2
        assert cfg.y == 20
        cfg.y = 42  # overwrite lambda with plain value
        assert cfg.y == 42

    def test_overwrite_value_with_lambda(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = 42
        cfg.y = lambda: cfg.x * 2  # overwrite plain value with lambda
        assert cfg.y == 20


# ── YAML warnings ────────────────────────────────────────

class TestYamlWarnings:
    def test_to_yaml_warns_on_callables(self, caplog):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2

        with caplog.at_level("WARNING", logger="xpconf"):
            cfg.to_yaml()

        assert "1 callable(s)" in caplog.text
        assert "y" in caplog.text
        assert "lossless" in caplog.text

    def test_save_yaml_warns_on_callables(self, caplog):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            with caplog.at_level("WARNING", logger="xpconf"):
                cfg.save_yaml(path)
            assert "1 callable(s)" in caplog.text
        finally:
            os.unlink(path)

    def test_no_warning_without_callables(self, caplog):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = 20

        with caplog.at_level("WARNING", logger="xpconf"):
            cfg.to_yaml()

        assert caplog.text == ""

    def test_warns_with_nested_callable_paths(self, caplog):
        cfg = ConfigDict()
        cfg.hidden_dim = 768
        cfg.model.mlp_dim = lambda: cfg.hidden_dim * 4
        cfg.model.head.input_dim = lambda: cfg.hidden_dim

        with caplog.at_level("WARNING", logger="xpconf"):
            cfg.to_yaml()

        assert "2 callable(s)" in caplog.text
        assert "model.mlp_dim" in caplog.text
        assert "model.head.input_dim" in caplog.text

    def test_to_dict_does_not_warn(self, caplog):
        """Plain to_dict should not emit warnings — it's not YAML-specific."""
        cfg = ConfigDict()
        cfg.x = 10
        cfg.y = lambda: cfg.x * 2

        with caplog.at_level("WARNING", logger="xpconf"):
            cfg.to_dict()

        assert caplog.text == ""


# ── Dotpath access ────────────────────────────────────────

class TestDotpath:
    def test_get_by_dotpath(self):
        cfg = ConfigDict()
        cfg.model.backbone.hidden_dim = 768
        assert cfg.get_by_dotpath("model.backbone.hidden_dim") == 768

    def test_get_by_dotpath_single_key(self):
        cfg = ConfigDict()
        cfg.x = 10
        assert cfg.get_by_dotpath("x") == 10

    def test_get_by_dotpath_missing_raises(self):
        cfg = ConfigDict(auto_nest=False)
        cfg.x = 10
        with pytest.raises((AttributeError, KeyError, TypeError)):
            cfg.get_by_dotpath("x.y.z")

    def test_set_by_dotpath(self):
        cfg = ConfigDict()
        cfg.model.backbone.hidden_dim = 768
        cfg.set_by_dotpath("model.backbone.hidden_dim", 1024)
        assert cfg.model.backbone.hidden_dim == 1024

    def test_set_by_dotpath_single_key(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.set_by_dotpath("x", 20)
        assert cfg.x == 20

    def test_set_by_dotpath_coerce_int(self):
        cfg = ConfigDict()
        cfg.model.layers = 12
        cfg.set_by_dotpath("model.layers", "24", coerce=True)
        assert cfg.model.layers == 24
        assert isinstance(cfg.model.layers, int)

    def test_set_by_dotpath_coerce_float(self):
        cfg = ConfigDict()
        cfg.lr = 3e-4
        cfg.set_by_dotpath("lr", "1e-5", coerce=True)
        assert cfg.lr == 1e-5
        assert isinstance(cfg.lr, float)

    def test_set_by_dotpath_coerce_bool_true(self):
        cfg = ConfigDict()
        cfg.use_bias = False
        for truthy in ("true", "True", "1", "yes"):
            cfg.set_by_dotpath("use_bias", truthy, coerce=True)
            assert cfg.use_bias is True

    def test_set_by_dotpath_coerce_bool_false(self):
        cfg = ConfigDict()
        cfg.use_bias = True
        for falsy in ("false", "False", "0", "no"):
            cfg.set_by_dotpath("use_bias", falsy, coerce=True)
            assert cfg.use_bias is False

    def test_set_by_dotpath_coerce_str(self):
        cfg = ConfigDict()
        cfg.name = "adam"
        cfg.set_by_dotpath("name", "sgd", coerce=True)
        assert cfg.name == "sgd"

    def test_set_by_dotpath_no_coerce(self):
        cfg = ConfigDict()
        cfg.layers = 12
        cfg.set_by_dotpath("layers", "24", coerce=False)
        assert cfg.layers == "24"  # stays as string

    def test_set_by_dotpath_respects_freeze(self):
        cfg = ConfigDict()
        cfg.x = 10
        cfg.freeze()
        with pytest.raises(FrozenError):
            cfg.set_by_dotpath("x", 20)

    def test_cli_pattern(self):
        """The intended 3-line CLI override pattern."""
        cfg = ConfigDict()
        cfg.model.hidden_dim = 768
        cfg.train.lr = 3e-4
        cfg.train.epochs = 100

        argv = ["model.hidden_dim=1024", "train.lr=1e-5", "train.epochs=50"]
        for arg in argv:
            key, value = arg.split("=", 1)
            cfg.set_by_dotpath(key, value, coerce=True)

        assert cfg.model.hidden_dim == 1024
        assert cfg.train.lr == 1e-5
        assert cfg.train.epochs == 50

