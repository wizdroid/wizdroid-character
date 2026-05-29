"""Validate all nodes: import, INPUT_TYPES, and method signature consistency."""

from typing import Any, Dict
import inspect
import pytest
from .conftest import import_node_module, ALL_NODE_MODULES


@pytest.mark.parametrize("module_name", ALL_NODE_MODULES)
class TestNodeImport:
    """Every node module must import without errors and export valid mappings."""

    def test_imports_cleanly(self, module_name):
        module = import_node_module(module_name)
        assert module is not None

    def test_has_class_mappings(self, module_name):
        module = import_node_module(module_name)
        # Accept NODE_CLASS_MAPPINGS or variant names
        for attr in ("NODE_CLASS_MAPPINGS", "META_PROMPT_NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, attr, None)
            if mappings:
                for cls_name, cls_obj in mappings.items():
                    assert hasattr(cls_obj, "INPUT_TYPES"), f"{cls_name} missing INPUT_TYPES"
                    assert hasattr(cls_obj, "RETURN_TYPES"), f"{cls_name} missing RETURN_TYPES"
                    assert hasattr(cls_obj, "FUNCTION"), f"{cls_name} missing FUNCTION"
                    assert hasattr(cls_obj, "CATEGORY"), f"{cls_name} missing CATEGORY"

    def test_input_types_valid(self, module_name):
        """INPUT_TYPES must return valid required/optional dicts with proper widget types."""
        module = import_node_module(module_name)
        for attr in ("NODE_CLASS_MAPPINGS", "META_PROMPT_NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, attr, None)
            if not mappings:
                continue
            for cls_name, cls_obj in mappings.items():
                try:
                    types = cls_obj.INPUT_TYPES()
                except Exception as e:
                    pytest.fail(f"{cls_name}.INPUT_TYPES() raised: {e}")
                assert "required" in types, f"{cls_name} missing 'required'"
                # Check widget type tuples have correct structure
                for key, spec in types["required"].items():
                    # ComfyUI IMAGE type can be a single-element tuple like ('IMAGE',)
                    if isinstance(spec, tuple) and len(spec) == 1:
                        continue
                    assert isinstance(spec, (list, tuple)) and len(spec) >= 2, \
                        f"{cls_name}.{key}: spec must be (type, config), got {spec}"
                    widget_type, config = spec[0], spec[1]
                    assert isinstance(config, dict), f"{cls_name}.{key}: config must be dict"

    def test_function_signature_matches_widgets(self, module_name):
        """The FUNCTION's parameter names should match INPUT_TYPES keys."""
        module = import_node_module(module_name)
        for attr in ("NODE_CLASS_MAPPINGS", "META_PROMPT_NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, attr, None)
            if not mappings:
                continue
            for cls_name, cls_obj in mappings.items():
                func_name = cls_obj.FUNCTION
                func = getattr(cls_obj, func_name)
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                # Remove 'self'
                if "self" in param_names:
                    param_names.remove("self")

                types = cls_obj.INPUT_TYPES()
                required_keys = list(types.get("required", {}).keys())
                optional_keys = list(types.get("optional", {}).keys())
                all_keys = required_keys + optional_keys

                # Each widget key should have a matching parameter
                for key in all_keys:
                    assert key in param_names, \
                        f"{cls_name}: INPUT_TYPES has '{key}' but function param missing it"

    def test_return_types_match(self, module_name):
        """RETURN_TYPES length should match function return value structure."""
        module = import_node_module(module_name)
        for attr in ("NODE_CLASS_MAPPINGS", "META_PROMPT_NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, attr, None)
            if not mappings:
                continue
            for cls_name, cls_obj in mappings.items():
                ret_types = cls_obj.RETURN_TYPES
                ret_names = getattr(cls_obj, "RETURN_NAMES", ret_types)
                assert len(ret_names) == len(ret_types), \
                    f"{cls_name}: RETURN_NAMES length != RETURN_TYPES length"


@pytest.mark.parametrize("module_name", ALL_NODE_MODULES)
class TestNodeDefaultValues:
    """Ensure default values in method signatures are valid."""

    def test_no_required_follows_optional(self, module_name):
        """No parameter with a default should be followed by one without."""
        module = import_node_module(module_name)
        for attr in ("NODE_CLASS_MAPPINGS", "META_PROMPT_NODE_CLASS_MAPPINGS"):
            mappings = getattr(module, attr, None)
            if not mappings:
                continue
            for cls_name, cls_obj in mappings.items():
                func_name = cls_obj.FUNCTION
                func = getattr(cls_obj, func_name)
                sig = inspect.signature(func)
                params = list(sig.parameters.values())
                # Skip 'self'
                params = [p for p in params if p.name != "self"]
                seen_default = False
                for p in params:
                    if p.default is not inspect.Parameter.empty:
                        seen_default = True
                    elif seen_default:
                        pytest.fail(
                            f"{cls_name}.{func_name}: '{p.name}' has no default "
                            f"but follows parameters with defaults"
                        )
