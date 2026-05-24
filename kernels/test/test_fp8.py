import pytest
import torch
from quartet2.fp8 import (
    fp32_to_fp8e4nv,
    fp32_to_fp8e5,
    fp8e4nv_to_fp32,
    fp8e5_to_fp32,
    fp32_fp8e4nv_fq,
    fp32_fp8e5_fq,
)

torch.random.manual_seed(42)


def _is_nan(x):
    return x != x


# ── E4M3 encode (fp32 → fp8) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x38),
        (1.5, 0x3C),
        (2.0, 0x40),
        (-1.0, 0xB8),
        (-2.0, 0xC0),
        (50.0, 0x64),
        (-5.0, 0xCA),
        (448.0, 0x7E),
        (-448.0, 0xFE),
        (240.0, 0x77),
        (-240.0, 0xF7),
        (float("inf"), 0x7E),
        (float("-inf"), 0xFE),
        (float("nan"), 0x7F),
    ],
)
def test_fp32_to_fp8_e4m3(x, expected):
    t = torch.tensor([x], device="cuda", dtype=torch.float32)
    out = fp32_to_fp8e4nv(t)
    if _is_nan(x):
        assert out[0].item() == 0x7F
    else:
        assert out[0].item() == expected, f"{x} → {hex(out[0].item())}, expected {hex(expected)}"


# ── E4M3 decode (fp8 → fp32) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "x, expected",
    [
        (0x00, 0.0),
        (0x80, -0.0),
        (0x38, 1.0),
        (0x3C, 1.5),
        (0x40, 2.0),
        (0xB8, -1.0),
        (0xC0, -2.0),
        (0x64, 48.0),
        (0xCA, -5.0),
        (0x7E, 448.0),
        (0xFE, -448.0),
        (0x77, 240.0),
        (0xF7, -240.0),
        (0x01, 2.0 ** -9),
        (0x04, 2.0 ** -7),
        (0x7F, float("nan")),
    ],
)
def test_fp8_to_fp32_e4m3(x, expected):
    t = torch.tensor([x], device="cuda", dtype=torch.uint8)
    out = fp8e4nv_to_fp32(t)
    if _is_nan(expected):
        assert _is_nan(out[0].item())
    else:
        assert out[0].item() == pytest.approx(expected, abs=1e-7)


# ── E5M2 encode (fp32 → fp8) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0x00),
        (-0.0, 0x80),
        (1.0, 0x3C),
        (1.5, 0x3E),
        (2.0, 0x40),
        (-1.0, 0xBC),
        (-2.0, 0xC0),
        (57344.0, 0x7B),
        (-57344.0, 0xFB),
        (float("inf"), 0x7B),
        (float("-inf"), 0xFB),
        (float("nan"), 0x7F),
    ],
)
def test_fp32_to_fp8_e5m2(x, expected):
    t = torch.tensor([x], device="cuda", dtype=torch.float32)
    out = fp32_to_fp8e5(t)
    if _is_nan(x):
        assert out[0].item() == 0x7F
    else:
        assert out[0].item() == expected, f"{x} → {hex(out[0].item())}, expected {hex(expected)}"


# ── E5M2 decode (fp8 → fp32) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "x, expected",
    [
        (0x00, 0.0),
        (0x80, -0.0),
        (0x3C, 1.0),
        (0x3E, 1.5),
        (0x40, 2.0),
        (0xBC, -1.0),
        (0xC0, -2.0),
        (0x7B, 57344.0),
        (0xFB, -57344.0),
        (0x7C, float("inf")),
        (0xFC, float("-inf")),
        (0x7D, float("nan")),
        (0x7E, float("nan")),
        (0x7F, float("nan")),
        (0x01, 2.0 ** -16),
        (0x04, 2.0 ** -14),
    ],
)
def test_fp8_to_fp32_e5m2(x, expected):
    t = torch.tensor([x], device="cuda", dtype=torch.uint8)
    out = fp8e5_to_fp32(t)
    if _is_nan(expected):
        assert _is_nan(out[0].item())
    elif expected == float("inf"):
        assert out[0].item() == float("inf")
    elif expected == float("-inf"):
        assert out[0].item() == float("-inf")
    else:
        assert out[0].item() == pytest.approx(expected, abs=1e-7)


# ── Fake quant equivalence: fused roundtrip = encode then decode ─────────────

@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
def test_fake_quant_equals_separate(dtype):
    x = torch.randn(256, device="cuda", dtype=torch.float32)
    if dtype == "e4m3":
        fq = fp32_fp8e4nv_fq(x)
        separate = fp8e4nv_to_fp32(fp32_to_fp8e4nv(x))
    else:
        fq = fp32_fp8e5_fq(x)
        separate = fp8e5_to_fp32(fp32_to_fp8e5(x))
    torch.testing.assert_close(fq, separate)


# ── Fake quant bitwidth sanity (f32→fp8→f32 should be near expected bits) ───

@pytest.mark.parametrize(
    "decode_fn, expected_bits",
    [(fp32_fp8e4nv_fq, 4.0), (fp32_fp8e5_fq, 4.0)],
)
def test_fake_quant_bitwidth(decode_fn, expected_bits):
    x = torch.randn(4096, 2048, device="cuda", dtype=torch.float32)
    dq = decode_fn(x)
    mse = (x - dq).pow(2).mean()
    power = x.pow(2).mean()
    sqnr = power / mse
    eff_bits = 0.5 * torch.log2(sqnr)
    assert eff_bits > expected_bits, f"Effective bitwidth {eff_bits:.2f} < {expected_bits}"


# ── Roundtrip accuracy on finite values ──────────────────────────────────────

def test_e4m3_roundtrip_finite():
    x = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 2.0, 50.0, -5.0, 0.5, 0.0078125],
        device="cuda",
        dtype=torch.float32,
    )
    rt = fp8e4nv_to_fp32(fp32_to_fp8e4nv(x))
    for inp, out in zip(x.tolist(), rt.tolist()):
        err = abs(inp - out) / max(abs(inp), 1e-30)
        assert err < 0.5, f"Large relative error for {inp}: got {out}"


def test_e5m2_roundtrip_finite():
    x = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 2.0, 57344.0, -57344.0, 0.5, 2.0 ** -14],
        device="cuda",
        dtype=torch.float32,
    )
    rt = fp8e5_to_fp32(fp32_to_fp8e5(x))
    for inp, out in zip(x.tolist(), rt.tolist()):
        err = abs(inp - out) / max(abs(inp), 1e-30)
        assert err < 0.5, f"Large relative error for {inp}: got {out}"
