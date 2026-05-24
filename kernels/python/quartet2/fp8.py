from functools import partial
import torch
import triton.language as tl
import triton

VERSION = 9
print("VERSION", VERSION)


@triton.jit
def _isnan(flt):
    s = tl.cast(flt, tl.uint32, bitcast=True)
    return (s & 0x7FFFFFFF) > 0x7F800000


@triton.jit
def _isinf(flt):
    s = tl.cast(flt, tl.uint32, bitcast=True)
    return (s == 0x7F800000) | (s == 0xFF800000)


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
        FP8_MAX_FLT = 0x7E
    else:
        FP8_NUM_EXPONENT_BITS = 5
        FP8_NUM_MANTISSA_BITS = 2
        FP8_EXPONENT_BIAS = 15
        FP8_MAX_EXPONENT = 15
        FP8_MIN_EXPONENT = -14
        FP8_MAX_FLT = 0x7B

    FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1
    FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1

    kF8_NaN = 0x7F

    # Extract bits from fp32
    s = tl.cast(flt, tl.uint32, bitcast=True)

    sign = ((s >> 24) & 0x80).to(tl.uint8)
    exp_biased = (s >> FP32_NUM_MANTISSA_BITS) & 0xFF
    exp = exp_biased.to(tl.int32) - FP32_EXPONENT_BIAS
    mantissa = s & 0x7FFFFF

    # NaN detection: exponent all ones, mantissa non-zero
    is_nan = _isnan(flt)

    # Inf detection
    is_inf = _isinf(flt)

    # exp == -128 catches the case where (exp_biased - 127) wraps in int8
    is_exp_neg128 = exp == -128

    sticky_bit = tl.zeros(sign.shape, dtype=tl.int32)
    skip_sign = tl.zeros(sign.shape, dtype=tl.int8)
    u = tl.zeros(sign.shape, dtype=tl.uint8)

    # Working mantissa that handles path-specific modifications
    mant_working = mantissa

    is_normal = (exp >= FP8_MIN_EXPONENT) & (exp <= FP8_MAX_EXPONENT)
    is_underflow = exp < FP8_MIN_EXPONENT
    is_overflow = exp > FP8_MAX_EXPONENT

    # Path A: Normal fp32 -> normal fp8
    # exp is in [FP8_MIN_EXPONENT, FP8_MAX_EXPONENT]

    exp_normal = exp + FP8_EXPONENT_BIAS
    u_normal = (
        (exp_normal.to(tl.uint32) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS
    ).to(tl.uint8)
    u_normal = u_normal | (
        mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
    ).to(tl.uint8)
    u = tl.where(is_normal, u_normal, u)

    # Path B: Underflow (exp < FP8_MIN_EXPONENT)

    rshift = (FP8_MIN_EXPONENT - exp).to(tl.int32)
    rshift_ok = rshift < FP32_NUM_BITS

    mant_under = mantissa | (1 << FP32_NUM_MANTISSA_BITS)

    sticky_bit = tl.where(
        is_underflow & rshift_ok,
        ((mant_under & ((1 << rshift) - 1)) != 0).to(tl.int32),
        sticky_bit,
    )

    mant_under = mant_under >> rshift
    mant_under = tl.where(
        is_underflow & rshift_ok,
        mant_under,
        tl.where(is_underflow, tl.zeros_like(mant_under), mantissa),
    )

    u_under = (
        (mant_under >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS))
        & FP8_MANTISSA_MASK
    ).to(tl.uint8)

    u = tl.where(is_underflow, u_under, u)
    mant_working = tl.where(is_underflow, mant_under, mant_working)

    # Path C: Overflow (exp > FP8_MAX_EXPONENT)
    is_overflow_exact = exp == (FP8_MAX_EXPONENT + 1)

    mantissa_tmp = (mantissa >> (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)).to(
        tl.uint8
    )
    mantissa_fits = mantissa_tmp < FP8_MANTISSA_MASK

    exp_overflow = exp + FP8_EXPONENT_BIAS
    u_overflow = (exp_overflow.to(tl.uint32) << FP8_NUM_MANTISSA_BITS).to(
        tl.uint8
    ) | mantissa_tmp

    u = tl.where(is_overflow & is_overflow_exact & mantissa_fits, u_overflow, u)

    may_be_nan = (
        is_overflow
        & is_overflow_exact
        & mantissa_fits
        & (mantissa_tmp == (FP8_MANTISSA_MASK - 1))
    )

    # Overflow saturation (quick return, bypasses rounding)
    is_overflow_sat = is_overflow & (~is_overflow_exact | ~mantissa_fits)

    # Apply rounding to u (for paths that go through rounding)
    NUM_BITS_SHIFT = FP32_NUM_MANTISSA_BITS - (FP8_NUM_MANTISSA_BITS + 1)
    round_bit = (mant_working >> NUM_BITS_SHIFT) & 1
    sticky_bit |= ((mant_working & ((1 << NUM_BITS_SHIFT) - 1)) != 0).to(tl.int32)

    do_round = ((round_bit & sticky_bit) | (round_bit & (u & 1))) != 0
    u_rounded = u + do_round.to(tl.uint8)

    skip_sign = tl.where(
        may_be_nan & do_round, tl.full(skip_sign.shape, 1, skip_sign.dtype), skip_sign
    )

    # Saturation
    u_rounded = tl.where(
        u_rounded > FP8_MAX_FLT, (sign | FP8_MAX_FLT).to(tl.uint8), u_rounded
    )

    # Apply sign
    u_final = tl.where(skip_sign == 0, (u_rounded | sign), u_rounded)

    # Early exit overrides (last = highest priority)
    result = tl.where(
        is_inf | is_exp_neg128 | is_overflow_sat,
        (sign | FP8_MAX_FLT).to(tl.uint8),
        u_final,
    )
    result = tl.where(is_nan, tl.full(u_final.shape, kF8_NaN, tl.uint8), result)

    return result


@triton.jit
def _fp32_to_fp8(flt, out, n, is_e4m3):
    n_range = tl.arange(0, 32)
    for i in range(0, n, 32):
        cur_range = n_range + i
        val = _fp32_to_fp8_impl(
            tl.load(flt + cur_range, mask=cur_range < n), IS_E4M3=is_e4m3
        )
        tl.store(out + cur_range, val, mask=cur_range < n)


def fp32_to_fp8e4nv(tens: torch.Tensor):
    assert tens.is_contiguous() and tens.is_cuda
    out = torch.empty(tens.shape, dtype=torch.uint8, device="cuda")
    _fp32_to_fp8e4nv[(1,)](tens, out, tens.numel())
    return out


@triton.jit
def _fp32_to_fp8e5m2(flt, out):
    val = _fp32_to_fp8_impl(tl.load(flt), IS_E4M3=False)
    tl.store(out, val)


@triton.jit
def _fp8_to_fp32_impl(x, IS_E4M3: tl.constexpr):
    FP32_NUM_BITS = 32
    FP32_NUM_EXPONENT_BITS = 8
    FP32_NUM_MANTISSA_BITS = 23
    FP32_EXPONENT_BIAS = 127
    FP32_INFINITY_MASK = 0x7F800000

    if IS_E4M3:
        FP8_NUM_EXPONENT_BITS = 4
        FP8_NUM_MANTISSA_BITS = 3
        FP8_EXPONENT_BIAS = 7
        FP8_MAX_EXPONENT = 7
        FP8_MAX_FLT = 0x7E
    else:
        FP8_NUM_EXPONENT_BITS = 5
        FP8_NUM_MANTISSA_BITS = 2
        FP8_EXPONENT_BIAS = 15
        FP8_MAX_EXPONENT = 15
        FP8_MAX_FLT = 0x7B

    FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1
    FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1
    FP8_NUM_BITS = 8
    kF32_NaN = 0x7FFFFFFF

    sign = (x >> (FP8_NUM_BITS - 1)) & 1
    exp = (x >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK
    mantissa = x & FP8_MANTISSA_MASK
    f = sign.to(tl.uint32) << (FP32_NUM_BITS - 1)

    # E4M3 specific: check for NaN pattern (exp == 15, mantissa == 0x7)
    is_e4m3_nan = (
        tl.full(x.shape, 0, tl.int8) if not IS_E4M3 else (exp == 15) & (mantissa == 0x7)
    )
    nan_result = tl.full(f.shape, kF32_NaN, tl.uint32)

    # Normal numbers: exp > 0
    is_normal = exp > 0
    if IS_E4M3:
        is_normal = is_normal
    else:
        # For E5M2, also need exp < 31 (max biased)
        is_normal = is_normal & (exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))

    exp_normal = (exp + (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS)).to(tl.uint32)
    f_normal = (
        f
        | (exp_normal << FP32_NUM_MANTISSA_BITS)
        | (mantissa.to(tl.uint32) << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS))
    )

    # Subnormal: exp == 0 && mantissa != 0
    is_subnormal = (exp == 0) & (mantissa != 0)

    # Normalize: find leading 1 in mantissa
    if IS_E4M3:
        mant_lz = tl.where(mantissa >= 4, 0, tl.where(mantissa >= 2, 1, 2))
    else:
        mant_lz = tl.where(mantissa >= 2, 0, 1)
    mantissa_sub = (mantissa << mant_lz) & FP8_MANTISSA_MASK
    exp_sub = (exp + (FP32_EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1 - mant_lz).to(
        tl.uint32
    )
    f_subnormal = (
        f
        | (exp_sub << FP32_NUM_MANTISSA_BITS)
        | (
            mantissa_sub.to(tl.uint32)
            << (FP32_NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)
        )
    )

    # Zero: exp == 0 && mantissa == 0 — sign-preserving zero (f already has sign)

    # Infinity/NaN from remaining exp patterns
    is_inf_fp8 = mantissa == 0
    f_inf = f | FP32_INFINITY_MASK
    f_nan_remaining = tl.full(f.shape, kF32_NaN, tl.uint32)

    # Combine: start with normal
    result = tl.where(is_e4m3_nan, nan_result, f_normal)

    # Subnormal override
    result = tl.where(is_subnormal, f_subnormal, result)

    # Zero (exp == 0 && mantissa == 0) — f already has sign bit, just keep it
    # No override needed for zero case

    # NaN (E4M3 case handled above; remaining overflow cases)
    is_overflow_exp = exp > 0
    if IS_E4M3:
        is_overflow_exp = is_overflow_exp & (exp == 15) & (mantissa != 0x7)
    else:
        is_overflow_exp = exp == (
            FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1
        )  # exp == 31 for E5M2
    result = tl.where(is_overflow_exp & is_inf_fp8, f_inf, result)
    result = tl.where(is_overflow_exp & (~is_inf_fp8), f_nan_remaining, result)

    return tl.cast(result, tl.float32, bitcast=True)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def elementwise_template(triton_fn, x, n, out, BLOCK_SIZE: tl.constexpr):
    cur_range = tl.arange(0, BLOCK_SIZE) + tl.program_id(axis=0) * BLOCK_SIZE
    val = triton_fn(tl.load(x + cur_range, mask=cur_range < n))
    tl.store(out + cur_range, val, mask=cur_range < n)


def fp32_to_fp8e4nv(tens):
    tens = tens.contiguous()
    out = torch.empty(tens.shape, dtype=torch.uint8, device="cuda")
    grid = lambda meta: (triton.cdiv(tens.numel(), meta["BLOCK_SIZE"]),)

    @triton.jit
    def _fp32_to_fp8e4nv_impl(flt):
        return _fp32_to_fp8_impl(flt, True)

    elementwise_template[grid](_fp32_to_fp8e4nv_impl, tens, tens.numel(), out)
    return out

def fp8e4nv_to_fp32(tens):
    tens = tens.contiguous()
    out = torch.empty(tens.shape, dtype=torch.float32, device="cuda")
    grid = lambda meta: (triton.cdiv(tens.numel(), meta["BLOCK_SIZE"]),)

    @triton.jit
    def _fp8e4nv_to_fp32_impl(flt):
        return _fp8_to_fp32_impl(flt, IS_E4M3=True)

    elementwise_template[grid](_fp8e4nv_to_fp32_impl, tens, tens.numel(), out)
    return out

@triton.jit
def fp8e5m2_to_fp32(x):
    return _fp8_to_fp32_impl(x, IS_E4M3=False)
