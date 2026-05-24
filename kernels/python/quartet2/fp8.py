import torch
import triton.language as tl
import triton

IS_E4M3 = True
FP32_NUM_BITS = 32
FP32_NUM_EXPONENT_BITS = 8
FP32_NUM_MANTISSA_BITS = 23
FP32_NAN = 0x7fffffff
FP32_INFINITY_MASK = 0x7f800000
FP32_MAX_EXPONENT  =  127
FP32_MIN_EXPONENT  = -126
FP32_EXPONENT_BIAS =  127

FP16_NUM_BITS = 16
FP16_NUM_EXPONENT_BITS = 5
FP16_NUM_MANTISSA_BITS = 10
FP16_NAN = 0x7fff
FP16_INFINITY_MASK = 0x7c00
FP16_MAX_EXPONENT  = 15
FP16_MIN_EXPONENT  = -14
FP16_EXPONENT_BIAS = 15

FP8_NUM_BITS = 8
FP8_NUM_EXPONENT_BITS = 4 if IS_E4M3 else 5
FP8_NUM_MANTISSA_BITS = 3 if IS_E4M3 else 2
FP8_NAN = 0x7f
FP8_INFINITY_MASK = 0x78 if IS_E4M3 else 0x7c
FP8_MAX_EXPONENT  =  7 if IS_E4M3 else  15
FP8_MIN_EXPONENT  = -6 if IS_E4M3 else -14
FP8_EXPONENT_BIAS =  7 if IS_E4M3 else  15

FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1
FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1

FP8_MAX_FLT = (0x7e if IS_E4M3 else 0x7b)
FP8_SAT_VAL_FP32 = 0x43800000
kF8_NaN = 0x7f

VERSION=6
print("VERSION", VERSION)

@triton.jit
def _fp32_to_fp8_impl(flt, IS_E4M3: tl.constexpr):
    # NaN -> fp8 NaN
    # +-inf -> +- fp8 max
    # exp==128 (127->overflow to -128) -> +-fp8 max
    # exp > max exp and ((exp == max exp and mantissa_tmp >= mask) or exp != max exp) -> +- fp8 max
    # Otherwise follow code path

    # FP32 constants
    FP32_NUM_MANTISSA_BITS = 23
    FP32_EXPONENT_BIAS = 127
    FP32_NUM_BITS = 32

    # FP8 constants determined by IS_E4M3 at compile time
    if IS_E4M3:
        FP8_NUM_EXPONENT_BITS = 4
        FP8_NUM_MANTISSA_BITS = 3
        FP8_EXPONENT_BIAS = 7
        FP8_MAX_EXPONENT = 7
        FP8_MIN_EXPONENT = -6
        FP8_MAX_FLT = 0x7e
    else:
        FP8_NUM_EXPONENT_BITS = 5
        FP8_NUM_MANTISSA_BITS = 2
        FP8_EXPONENT_BIAS = 15
        FP8_MAX_EXPONENT = 15
        FP8_MIN_EXPONENT = -14
        FP8_MAX_FLT = 0x7b

    FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1
    FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1

    kF8_NaN = 0x7f

    # Extract bits from fp32
    s = tl.cast(flt, tl.uint32, bitcast=True)

    sign = ((s >> 24) & 0x80).to(tl.uint8)
    exp_biased = (s >> FP32_NUM_MANTISSA_BITS) & 0xFF
    exp = exp_biased - FP32_EXPONENT_BIAS
    mantissa = s & 0x7FFFFF

    # NaN detection: exponent all ones, mantissa non-zero
    is_nan = flt == float("nan")

    # Inf detection
    is_inf = flt == float("inf") or flt == -float("inf")

    # exp == -128 catches the case where (exp_biased - 127) wraps in int8
    is_exp_neg128 = exp == -128

    sticky_bit = tl.zeros(sign.shape, dtype=tl.int32)
    skip_sign = tl.zeros(sign.shape, dtype=tl.int8)
    u = tl.zeros(sign.shape, dtype=tl.uint8)
    sticky_path = tl.zeros_like(sticky_bit)

    # Working mantissa that handles path-specific modifications
    mant_working = mantissa

    is_normal = (exp >= FP8_MIN_EXPONENT) and (exp <= FP8_MAX_EXPONENT)
    is_underflow = (exp < FP8_MIN_EXPONENT)
    is_overflow = not is_normal and not is_underflow

    # Path A: Normal fp32 -> normal fp8
    # exp is in [FP8_MIN_EXPONENT, FP8_MAX_EXPONENT]

    exp_normal = exp + FP8_EXPONENT_BIAS
    u_normal = ((exp_normal.to(tl.uint32) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS).to(tl.uint8)
    u_normal = u_normal | (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)).to(tl.uint8)

    # Path B: Underflow (exp < FP8_MIN_EXPONENT)

    rshift = (FP8_MIN_EXPONENT - exp).to(tl.int32)
    rshift_ok = rshift < FP32_NUM_BITS

    mant_under = mantissa | (1 << FP32_NUM_MANTISSA_BITS)

    sticky_bit = tl.where(is_underflow and rshift_ok,
        ((mant_under & ((1 << rshift) - 1)) != 0).to(tl.int32),
        0)

    mant_under = mant_under >> rshift
    mant_under = tl.where(is_underflow and rshift_ok, mant_under, 0)

    u_under = ((mant_under >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK).to(tl.uint8)

    # For rshift >= FP32_NUM_BITS, flush to zero
    u_under = tl.where(is_underflow and rshift_ok, u_under, 0)
    mant_under = tl.where(is_underflow, mant_under, mantissa)

    u = tl.where(is_underflow, u_under, u)
    mant_working = tl.where(is_underflow, mant_under, mant_working)

    # Path C: Overflow (exp > FP8_MAX_EXPONENT)
    is_overflow_exact = (exp == (FP8_MAX_EXPONENT + 1))

    mantissa_tmp = (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)).to(tl.uint8)
    mantissa_fits = mantissa_tmp < FP8_MANTISSA_MASK

    exp_overflow = exp + FP8_EXPONENT_BIAS
    u_overflow = (exp_overflow.to(tl.uint32) << FP8_NUM_MANTISSA_BITS).to(tl.uint8) | mantissa_tmp

    u = tl.where(is_overflow and is_overflow_exact and mantissa_fits, u_overflow, u)

    may_be_nan = is_overflow and is_overflow_exact and mantissa_fits and mantissa_tmp == (FP8_MANTISSA_MASK - 1)

    # Overflow saturation (quick return, bypasses rounding)
    is_overflow_sat = is_overflow and (not is_overflow_exact or not mantissa_fits)

    # Apply rounding to u (for paths that go through rounding)
    NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1)
    round_bit = ((mant_working >> NUM_BITS_SHIFT) & 1)
    sticky_bit |= ((mant_working & ((1 << NUM_BITS_SHIFT) - 1)) != 0).to(tl.int32)

    do_round = ((round_bit & sticky_bit) | (round_bit & (u & 1))) != 0
    u_rounded = (u + do_round.to(tl.uint8))

    skip_sign = tl.where(may_be_nan and do_round, tl.full(skip_sign.shape, 1, skip_sign.dtype), skip_sign)

    # Saturation
    u_rounded = tl.where(u_rounded > FP8_MAX_FLT,
        (sign | FP8_MAX_FLT).to(tl.uint8),
        u_rounded)

    # Apply sign
    u_final = tl.where(skip_sign == 0, (u_rounded | sign), u_rounded)

    # Early exit overrides for NaN, Inf, exp==-128, overflow-sat
    result = tl.where(is_nan, tl.full(u_final.shape, kF8_NaN, tl.uint8), u_final)
    result = tl.where(is_inf or is_exp_neg128 or is_overflow_sat, (sign | FP8_MAX_FLT).to(tl.uint8), result)

    return result


@triton.jit
def _fp32_to_fp8e4nv(flt, out):
    val = _fp32_to_fp8_impl(tl.load(flt), IS_E4M3=True)
    tl.store(out, val)


@triton.jit
def _fp32_to_fp8e5m2(flt, out):
    val = _fp32_to_fp8_impl(tl.load(flt), IS_E4M3=False)
    tl.store(out, val)

def fp32_to_fp8e4nv(tens):
    out = torch.empty_like(tens)
    _fp32_to_fp8e4nv[(1,)](tens, out)
    return out


@triton.jit
def _fp8_to_fp32_impl(x, IS_E4M3: tl.constexpr):
    FP32_NUM_BITS = 32
    FP32_NUM_EXPONENT_BITS = 8
    FP32_NUM_MANTISSA_BITS = 23
    FP32_EXPONENT_BIAS = 127
    FP32_INFINITY_MASK = 0x7f800000

    if IS_E4M3:
        FP8_NUM_EXPONENT_BITS = 4
        FP8_NUM_MANTISSA_BITS = 3
        FP8_EXPONENT_BIAS = 7
        FP8_MAX_EXPONENT = 7
        FP8_MAX_FLT = 0x7e
    else:
        FP8_NUM_EXPONENT_BITS = 5
        FP8_NUM_MANTISSA_BITS = 2
        FP8_EXPONENT_BIAS = 15
        FP8_MAX_EXPONENT = 15
        FP8_MAX_FLT = 0x7b

    FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1
    FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1
    kF32_NaN = 0x7fffffff

    sign = (x >> (FP8_NUM_BITS - 1)) & 1
    exp = (x >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK
    mantissa = x & FP8_MANTISSA_MASK
    f = (sign.to(tl.uint32) << (FP32_NUM_BITS - 1))

    # E4M3 specific: check for NaN pattern (exp == 15, mantissa == 0x7)
    is_e4m3_nan = tl.full(x.shape, 0, tl.int8) if not IS_E4M3 else (exp == 15) & (mantissa == 0x7)
    nan_result = tl.full(f.shape, kF32_NaN, tl.uint32)

    # Normal numbers: exp > 0
    is_normal = (exp > 0)
    if IS_E4M3:
        is_normal = is_normal
    else:
        # For E5M2, also need exp < 31 (max biased)
        is_normal = is_normal & (exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))

    exp_normal = (exp + (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS)).to(tl.uint32)
    f_normal = f | (exp_normal << FP32_NUM_MANTISSA_BITS) | (mantissa.to(tl.uint32) << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS))

    # Subnormal: exp == 0 && mantissa != 0
    is_subnormal = (exp == 0) & (mantissa != 0)

    # Normalize: find leading 1 in mantissa
    if IS_E4M3:
        mant_lz = tl.where(mantissa >= 4, 0, tl.where(mantissa >= 2, 1, 2))
    else:
        mant_lz = tl.where(mantissa >= 2, 0, 1)
    mantissa_sub = (mantissa << mant_lz) & FP8_MANTISSA_MASK
    exp_sub = (exp + (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1 - mant_lz).to(tl.uint32)
    f_subnormal = f | (exp_sub << FP32_NUM_MANTISSA_BITS) | (mantissa_sub.to(tl.uint32) << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS))

    # Zero: exp == 0 && mantissa == 0 — sign-preserving zero (f already has sign)

    # Infinity/NaN from remaining exp patterns
    is_inf_fp8 = (mantissa == 0)
    f_inf = (f | FP32_INFINITY_MASK)
    f_nan_remaining = tl.full(f.shape, kF32_NaN, tl.uint32)

    # Combine: start with normal
    result = tl.where(is_e4m3_nan, nan_result, f_normal)

    # Subnormal override
    result = tl.where(is_subnormal, f_subnormal, result)

    # Zero (exp == 0 && mantissa == 0) — f already has sign bit, just keep it
    # No override needed for zero case

    # NaN (E4M3 case handled above; remaining overflow cases)
    is_overflow_exp = (exp > 0)
    if IS_E4M3:
        is_overflow_exp = is_overflow_exp & (exp == 15) & (mantissa != 0x7)
    else:
        is_overflow_exp = (exp == (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))  # exp == 31 for E5M2
    result = tl.where(is_overflow_exp and is_inf_fp8, f_inf, result)
    result = tl.where(is_overflow_exp and not is_inf_fp8, f_nan_remaining, result)

    return tl.cast(result, tl.float32, bitcast=True)


@triton.jit
def fp8e4nv_to_fp32(x):
    return _fp8_to_fp32_impl(x, IS_E4M3=True)


@triton.jit
def fp8e5m2_to_fp32(x):
    return _fp8_to_fp32_impl(x, IS_E4M3=False)
