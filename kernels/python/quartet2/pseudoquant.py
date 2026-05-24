from random import randint

import torch
torch.set_float32_matmul_precision('high')
import torch.nn.functional as F
import triton
import triton.language as tl
from .fp8 import _fp8_to_fp32_impl, _fp32_to_fp8_impl

@triton.jit
def rtn_fp4(x):
    x_abs = tl.abs(x)
    x_sign = tl.where(
        x > 0,
        1,
        -1,
    )
    x_fp4_abs = tl.where(
        x_abs >= 5,
        6,
        tl.where(
            x_abs >= 3.5,
            4,
            tl.where(
                x_abs >= 2.5,
                3,
                tl.where(
                    x_abs >= 1.75,
                    2,
                    tl.where(
                        x_abs >= 1.25,
                        1.5,
                        tl.where(
                            x_abs >= 0.75,
                            1,
                            tl.where(
                                x_abs >= 0.25,
                                0.5,
                                0.0,
                            )
                        )
                    )
                )
            )
        )
    )
    return x_fp4_abs * x_sign


@triton.jit
def get_scales(x, amax, val_max, scales_max):
    s_dec = tl.where(
        amax == 0.0,
        1.0,
        amax / scales_max / val_max,
    )
    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = _fp8_to_fp32_impl(_fp32_to_fp8_impl(s_dec_b / s_dec, True), True)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3, s_dec


@triton.jit
def get_alt_scales(x, val_max, s_dec):    
    s_dec_b = tl.max(tl.abs(x), axis=-1, keep_dims=True) / val_max
    s_dec_b_e4m3 = _fp8_to_fp32_impl(_fp32_to_fp8_impl(s_dec_b * (6/4) / s_dec, True), True)
    s_dec_b_e4m3 = tl.where(
        s_dec_b_e4m3 == 0,
        1.0,
        s_dec_b_e4m3,
    )
    return s_dec_b_e4m3


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def rtn_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    four_over_six: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):        
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask, other=0.0)  
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # amax
    scales_max = 256.00 if four_over_six else 448.00
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    x_dequantized = x_fp4 * (s_dec_b_e4m3 * s_dec)
    
    if not four_over_six:
        best_x_dequantized = x_dequantized
    else:
        alt_s_dec_b_e4m3 = get_alt_scales(x_grouped, val_max, s_dec)
        alt_x_scaled = x_grouped / (alt_s_dec_b_e4m3 * s_dec)
        
        alt_x_fp4 = rtn_fp4(alt_x_scaled)
        alt_x_dequantized = alt_x_fp4 * (alt_s_dec_b_e4m3 * s_dec)
        
        error_six = tl.sum((x_grouped - x_dequantized) * (x_grouped - x_dequantized), axis=-1, keep_dims=True)
        error_four = tl.sum((x_grouped - alt_x_dequantized) * (x_grouped - alt_x_dequantized), axis=-1, keep_dims=True)
        
        best_x_dequantized = tl.where(
            error_six <= error_four,
            x_dequantized,
            alt_x_dequantized,
        )
    
    x_dequantized_flat = tl.reshape(best_x_dequantized, (BLOCK_SIZE,))
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def rtn_1x16s_fp4_kernel_wrapper(
    x,
    scale_override: float,
    group_size: int,
    four_over_six: bool = False,
):
    x = x.contiguous()
    output = torch.empty_like(x)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    rtn_1x16s_fp4_kernel[grid](
        x_ptr=x,
        amax_ptr=x.abs().max(),
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        four_over_six=four_over_six,
    )
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def eden_1x16s_fp4_kernel(
    x_ptr,
    amax_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    scale_override: tl.constexpr,
    group_size: tl.constexpr,
    seed: int,
    BLOCK_SIZE: tl.constexpr,
):    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))

    # scale
    scales_max = 255.99
    val_max = 6.0 / scale_override
    amax = tl.load(amax_ptr)
    
    s_dec_b_e4m3, s_dec = get_scales(x_grouped, amax, val_max, scales_max)
    x_scaled = x_grouped / (s_dec_b_e4m3 * s_dec)
    
    # quantize
    x_fp4 = rtn_fp4(x_scaled)
    
    # Calculate EDEN scale    
    num = tl.sum(x_scaled * x_scaled, axis=-1, keep_dims=True)
    denom = tl.sum(x_scaled * x_fp4, axis=-1, keep_dims=True)
    
    correction = tl.where(
        denom == 0.0,
        1.0,
        num / denom,
    ) # [BLOCK_SIZE // group_size, 1]
    
    # Apply EDEN scale
    corrected_scales = s_dec_b_e4m3 * correction # [BLOCK_SIZE // group_size, 1]

    bitscales = _fp32_to_fp8_impl(corrected_scales, True)
    prevscale = _fp8_to_fp32_impl(bitscales - 1, True)
    currscale = _fp8_to_fp32_impl(bitscales, True)
    nextscale = _fp8_to_fp32_impl(bitscales + 1, True)
    
    up = tl.where(
        currscale > corrected_scales,
        currscale,
        nextscale,
    )
    down = tl.where(
        currscale > corrected_scales,
        prevscale,
        currscale,
    )
    
    prob_up = (corrected_scales - down) / (up - down)
    
    scale_start_idx = pid * (BLOCK_SIZE // group_size)
    scale_offsets = scale_start_idx + tl.arange(0, BLOCK_SIZE // group_size)
    sampled_prob = tl.rand(seed, scale_offsets).reshape(BLOCK_SIZE // group_size, 1)
    
    scales = tl.where(
        sampled_prob < prob_up,
        up,
        down,
    )
    scales = tl.reshape(scales, (BLOCK_SIZE // group_size, 1))
    x_fp4 = tl.reshape(x_fp4, (BLOCK_SIZE // group_size, group_size))
    
    # Reshape back to flat form for storage
    x_dequantized = x_fp4 * scales * s_dec
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)

@torch.compiler.disable()
def eden_1x16s_fp4_kernel_wrapper(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor,
    scale_override: float,
    group_size: int,
) -> [torch.Tensor, torch.Tensor]:
    hadamard_dim = hadamard_matrix.size(0)
    assert hadamard_matrix.size(1) == hadamard_dim
    assert x.size(-1) % hadamard_dim == 0
    assert hadamard_dim % group_size == 0

    x_had = (x.reshape(-1, hadamard_matrix.size(0)) @ hadamard_matrix.T).reshape_as(x).contiguous()
    amax = x_had.abs().max()

    output = torch.empty_like(x_had)
    seed = randint(0, 1000000)
    
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    
    eden_1x16s_fp4_kernel[grid](
        x_ptr=x_had,
        amax_ptr=amax,
        output_ptr=output,
        n_elements=n_elements,
        scale_override=scale_override,
        group_size=group_size,
        seed=seed,
    )
    return output


def rerotate_hadamard(hadamard_matrix):
    signs = torch.randint(
            0, 2, (hadamard_matrix.size(0),),
            device=hadamard_matrix.device,
            dtype=hadamard_matrix.dtype
        ) * 2 - 1
    return hadamard_matrix * signs[None, :] # NOTE: rerotate along last dim, inner dim for TN GEMM


class Quartet_II_pseudoquant_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, had, disable_backward_quant: bool, four_over_six: bool):
        ctx.orig_shape = input.shape
        
        flat_input = input.reshape(-1, input.shape[-1])
        
        ctx.batch = flat_input.shape[0]
        ctx.in_dim = weight.shape[1]
        ctx.out_dim = weight.shape[0]
        ctx.disable_backward_quant = disable_backward_quant
        
        autocast_enabled = torch.is_autocast_enabled("cuda")
        if autocast_enabled:
            flat_input = flat_input.to(torch.bfloat16)
            weight = weight.to(torch.bfloat16)

        assert flat_input.dtype == torch.bfloat16
        assert weight.dtype == torch.bfloat16
        
        forward_scale_override = 1.0
        
        input_fp4 = rtn_1x16s_fp4_kernel_wrapper(flat_input, scale_override=forward_scale_override, group_size=16, four_over_six=four_over_six)
        weight_fp4 = rtn_1x16s_fp4_kernel_wrapper(weight, scale_override=forward_scale_override, group_size=16, four_over_six=four_over_six)

        ctx.save_for_backward(input_fp4, weight_fp4, had)
        return F.linear(input_fp4, weight_fp4).reshape(ctx.orig_shape[:-1] + (ctx.out_dim,),)

    @staticmethod
    def backward(ctx, grad_output):
        # Load ctx and reshape
        input_fp4, weight_fp4, had = ctx.saved_tensors
        backward_scale_override = (17 / 16) * 0.93
        
        autocast_enabled = torch.is_autocast_enabled("cuda")
        if autocast_enabled:
            grad_output = grad_output.to(torch.bfloat16)
        
        input_fp4 = input_fp4.reshape(ctx.batch, ctx.in_dim)
        grad_output = grad_output.reshape(ctx.batch, ctx.out_dim)
        
        # Re-randomize the rotation
        had = rerotate_hadamard(had)
        
        # No backward quant if flag
        if ctx.disable_backward_quant:
            grad_input = F.linear(
                grad_output,
                weight_fp4.T,
                None,
            ).view(ctx.batch, ctx.in_dim)
            
            grad_weight = F.linear(
                grad_output.T,
                input_fp4.T,
                None,
            )
            return grad_input.reshape(ctx.orig_shape), grad_weight, None, None, None
        
        # EW
        e_ht_fp4 = eden_1x16s_fp4_kernel_wrapper(grad_output, had, backward_scale_override, 16)
        weight_tht_fp4 = eden_1x16s_fp4_kernel_wrapper(weight_fp4.T, had, backward_scale_override, 16)
        
        grad_input = F.linear(
            e_ht_fp4,
            weight_tht_fp4,
            None,
        ).view(ctx.batch, ctx.in_dim)

        # EtX
        e_tht_fp4 = eden_1x16s_fp4_kernel_wrapper(grad_output.T, had, backward_scale_override, 16)
        input_tht_fp4 = eden_1x16s_fp4_kernel_wrapper(input_fp4.T, had, backward_scale_override, 16)
        
        grad_weight = F.linear(
            e_tht_fp4,
            input_tht_fp4,
            None,
        )
        
        return grad_input.reshape(ctx.orig_shape), grad_weight, None, None, None
