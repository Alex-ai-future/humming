from types import SimpleNamespace

import pytest
import torch

from humming import dtypes
from humming.config import LayerConfig
from humming.config.mma import get_dtype_compatibility_error
from humming.schema import HummingInputSchema, HummingWeightSchema, is_humming_schema_compatible

ACTIVATION_DTYPES = (
    dtypes.float16,
    dtypes.bfloat16,
    dtypes.float8e4m3,
    dtypes.float8e5m2,
    dtypes.float8e3m4,
    dtypes.float4e2m1,
    dtypes.float4e0m3,
    dtypes.int8,
    dtypes.int4,
)
OUTPUT_DTYPES = (dtypes.float16, dtypes.bfloat16)


def _make_layer_config(a_dtype, c_dtype, bs_dtype):
    return LayerConfig(
        sm_version=90,
        shape_n=128,
        shape_k=512,
        a_dtype=a_dtype,
        b_dtype=dtypes.uint4,
        c_dtype=c_dtype,
        bs_dtype=bs_dtype,
    )


def _make_schemas(a_dtype, c_dtype, bs_dtype):
    input_schema = HummingInputSchema(a_dtype=a_dtype)
    weight_schema = HummingWeightSchema(b_dtype=dtypes.uint4, bs_dtype=bs_dtype)
    param_dtype = dtypes.torch_dtype_map[c_dtype]
    return input_schema, weight_schema, param_dtype


@pytest.mark.parametrize("a_dtype", ACTIVATION_DTYPES, ids=str)
@pytest.mark.parametrize("c_dtype", OUTPUT_DTYPES, ids=str)
def test_layer_config_accepts_only_supported_activation_output_pairs(a_dtype, c_dtype):
    bs_dtype = a_dtype if a_dtype.num_bits == 16 else dtypes.bfloat16
    is_supported = (
        a_dtype == c_dtype
        if a_dtype.num_bits == 16
        else a_dtype != dtypes.float8e5m2 or c_dtype == dtypes.bfloat16
    )

    if is_supported:
        config = _make_layer_config(a_dtype, c_dtype, bs_dtype)
        assert (config.a_dtype, config.c_dtype) == (a_dtype, c_dtype)
    else:
        with pytest.raises(ValueError) as error:
            _make_layer_config(a_dtype, c_dtype, bs_dtype)
        assert str(a_dtype) in str(error.value)
        assert str(c_dtype) in str(error.value)


def test_layer_config_rejects_non_output_dtype():
    expected_dtype = str(dtypes.float32)

    with pytest.raises(ValueError) as error:
        _make_layer_config(dtypes.float16, dtypes.float32, dtypes.float16)

    assert expected_dtype in str(error.value)


@pytest.mark.parametrize(
    ("a_dtype", "c_dtype", "bs_dtype", "is_supported"),
    [
        (dtypes.float16, dtypes.float16, dtypes.float16, True),
        (dtypes.float16, dtypes.float16, dtypes.bfloat16, False),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16, True),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.float16, False),
        (dtypes.float16, dtypes.float16, dtypes.float8e4m3, True),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.float8e8m0, True),
        (dtypes.float16, dtypes.float16, dtypes.float32, True),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.float32, True),
    ],
)
def test_layer_config_weight_scale_dtype_validation(a_dtype, c_dtype, bs_dtype, is_supported):
    if is_supported:
        config = _make_layer_config(a_dtype, c_dtype, bs_dtype)
        assert config.bs_dtype == bs_dtype
    else:
        with pytest.raises(ValueError) as error:
            _make_layer_config(a_dtype, c_dtype, bs_dtype)
        assert str(a_dtype) in str(error.value)
        assert str(bs_dtype) in str(error.value)


@pytest.mark.parametrize(
    ("a_dtype", "c_dtype", "bs_dtype", "is_supported"),
    [
        (dtypes.float16, dtypes.float16, dtypes.float16, True),
        (dtypes.float16, dtypes.bfloat16, dtypes.bfloat16, False),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.bfloat16, True),
        (dtypes.bfloat16, dtypes.float16, dtypes.float16, False),
        (dtypes.float8e4m3, dtypes.float16, dtypes.bfloat16, True),
        (dtypes.float8e5m2, dtypes.bfloat16, dtypes.bfloat16, True),
        (dtypes.float8e5m2, dtypes.float16, dtypes.float16, False),
        (dtypes.int8, dtypes.bfloat16, dtypes.bfloat16, True),
        (dtypes.float16, dtypes.float16, dtypes.bfloat16, False),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.float16, False),
        (dtypes.float16, dtypes.float16, dtypes.float8e4m3, True),
        (dtypes.bfloat16, dtypes.bfloat16, dtypes.float8e8m0, True),
        (dtypes.float16, dtypes.float16, dtypes.float32, True),
    ],
)
def test_layer_config_and_schema_use_same_dtype_rules(monkeypatch, a_dtype, c_dtype, bs_dtype, is_supported):
    monkeypatch.setattr("humming.schema.humming.current_device", SimpleNamespace(sm_version=90))
    input_schema, weight_schema, param_dtype = _make_schemas(a_dtype, c_dtype, bs_dtype)

    assert is_humming_schema_compatible(input_schema, weight_schema, param_dtype) is is_supported
    if is_supported:
        config = _make_layer_config(a_dtype, c_dtype, bs_dtype)
        assert get_dtype_compatibility_error(config.a_dtype, config.c_dtype, config.bs_dtype) is None
    else:
        with pytest.raises(ValueError):
            _make_layer_config(a_dtype, c_dtype, bs_dtype)


def test_schema_rejects_unsupported_param_dtype():
    input_schema = HummingInputSchema(a_dtype=dtypes.float16)
    weight_schema = HummingWeightSchema(b_dtype=dtypes.uint4)

    assert not is_humming_schema_compatible(input_schema, weight_schema, torch.float32)
