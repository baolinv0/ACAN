"""
Adjustment selection utilities.
Flatten adjustment dictionaries into vectors and restore them back.
"""

import json

import numpy as np


def _get_class_value(value, class_idx, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if class_idx < arr.shape[0]:
            return float(arr[class_idx])
        return default
    if arr.ndim == 2:
        if class_idx < arr.shape[0]:
            return float(arr[class_idx].mean())
        return default
    return default


def _get_class_coeff(value, class_idx, default, coeffs=12):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(value)
    if arr.ndim == 1:
        if arr.shape[0] == coeffs:
            return float(arr.mean())
        if class_idx < arr.shape[0]:
            return float(arr[class_idx])
        return default
    if arr.ndim == 2:
        if class_idx < arr.shape[0]:
            return float(arr[class_idx].mean())
        return default
    return default


def normalize_adjustments(adjustments):
    if adjustments is None:
        return {}
    if isinstance(adjustments, str):
        return json.loads(adjustments)
    return adjustments


def flatten_adjustments(
    adjustments,
    focus_class,
    include_hdrnet=True,
    include_local=True,
    coeffs=12,
):
    adjustments = normalize_adjustments(adjustments)
    base = adjustments.get("base", adjustments)
    global_adj = base.get("global", {})
    class_adj = base.get("class", {})
    local_adj = base.get("local", {})

    values = []
    names = []

    defaults = {
        "global_gain": 1.0,
        "global_gamma": 1.0,
        "global_white": 1.0,
        "global_bias": 0.0,
        "class_gain": 1.0,
        "class_bias": 0.0,
        "local_gain": 0.0,
        "local_bias": 0.0,
    }

    values.append(float(global_adj.get("gain", defaults["global_gain"])))
    names.append("global_gain")
    values.append(float(global_adj.get("gamma", defaults["global_gamma"])))
    names.append("global_gamma")
    values.append(float(global_adj.get("white", defaults["global_white"])))
    names.append("global_white")
    values.append(float(global_adj.get("bias", defaults["global_bias"])))
    names.append("global_bias")

    values.append(_get_class_value(class_adj.get("gain"), focus_class, defaults["class_gain"]))
    names.append("class_gain")
    values.append(_get_class_value(class_adj.get("bias"), focus_class, defaults["class_bias"]))
    names.append("class_bias")

    if include_local:
        values.append(
            _get_class_value(local_adj.get("gain_strength"), focus_class, defaults["local_gain"])
        )
        names.append("local_gain_strength")
        values.append(
            _get_class_value(local_adj.get("bias_strength"), focus_class, defaults["local_bias"])
        )
        names.append("local_bias_strength")

    if include_hdrnet:
        mix = adjustments.get("mix", None)
        mix_val = _get_class_value(mix, focus_class, 0.0)
        values.append(mix_val)
        names.append("hdrnet_mix")

        hdrnet = adjustments.get("hdrnet_semantic", {})
        values.append(
            _get_class_value(hdrnet.get("guide_bias"), focus_class, 0.0)
        )
        names.append("hdrnet_guide_bias")
        values.append(
            _get_class_coeff(hdrnet.get("coeff_scale"), focus_class, 1.0, coeffs=coeffs)
        )
        names.append("hdrnet_coeff_scale")
        values.append(
            _get_class_coeff(hdrnet.get("coeff_bias"), focus_class, 0.0, coeffs=coeffs)
        )
        names.append("hdrnet_coeff_bias")

    return np.asarray(values, dtype=np.float32), names


def vector_to_adjustments(
    vec,
    focus_class,
    num_classes,
    include_hdrnet=True,
    include_local=True,
    coeffs=12,
):
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    idx = 0

    def take():
        nonlocal idx
        val = float(vec[idx])
        idx += 1
        return val

    adjustments = {
        "base": {
            "global": {
                "gain": take(),
                "gamma": take(),
                "white": take(),
                "bias": take(),
            },
            "class": {},
            "local": {},
        }
    }
    class_gain = take()
    class_bias = take()

    gains = np.ones((num_classes,), dtype=np.float32)
    biases = np.zeros((num_classes,), dtype=np.float32)
    gains[int(focus_class)] = class_gain
    biases[int(focus_class)] = class_bias
    adjustments["base"]["class"]["gain"] = gains
    adjustments["base"]["class"]["bias"] = biases

    if include_local:
        local_gain = take()
        local_bias = take()
        local_gains = np.zeros((num_classes,), dtype=np.float32)
        local_biases = np.zeros((num_classes,), dtype=np.float32)
        local_gains[int(focus_class)] = local_gain
        local_biases[int(focus_class)] = local_bias
        adjustments["base"]["local"]["gain_strength"] = local_gains
        adjustments["base"]["local"]["bias_strength"] = local_biases

    if include_hdrnet:
        mix = take()
        guide_bias = take()
        coeff_scale = take()
        coeff_bias = take()

        mix_map = np.zeros((num_classes,), dtype=np.float32)
        mix_map[int(focus_class)] = mix
        adjustments["mix"] = mix_map

        coeff_scale_map = np.ones((num_classes, coeffs), dtype=np.float32)
        coeff_bias_map = np.zeros((num_classes, coeffs), dtype=np.float32)
        coeff_scale_map[int(focus_class)] = coeff_scale
        coeff_bias_map[int(focus_class)] = coeff_bias
        guide_bias_map = np.zeros((num_classes,), dtype=np.float32)
        guide_bias_map[int(focus_class)] = guide_bias
        adjustments["hdrnet_semantic"] = {
            "coeff_scale": coeff_scale_map,
            "coeff_bias": coeff_bias_map,
            "guide_bias": guide_bias_map,
        }

    return adjustments
