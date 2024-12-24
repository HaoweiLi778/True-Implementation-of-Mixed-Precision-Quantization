import torch

def quantize(x, input_compress_settings={}):
    """
    Quantizes the input tensor `x` to a specified bit width and stores it as a `torch.uint8` format.

    Args:
        x (torch.Tensor): Input tensor of floating-point type.
        input_compress_settings (dict): Compression parameters, including:
            - 'n' (int): Quantization bit width (supports 2-bit, 4-bit, and 8-bit).
    Returns:
        packed_x (torch.Tensor): Quantized tensor stored as uint8.
        scale (float): Scaling factor for dequantization.
        n (int): Quantization bit width.
    """
    compress_settings = {'n': 4}  # Default to 4-bit quantization
    compress_settings.update(input_compress_settings)
    n = compress_settings['n']

    if n not in [2, 4, 8]:
        raise ValueError("Only 2-bit, 4-bit, and 8-bit quantization are supported.")

    # Convert to float
    x = x.float()

    # Compute the maximum norm (used for scaling)
    x_norm = torch.norm(x, p=float('inf'))
    if x_norm == 0:
        return torch.zeros_like(x, dtype=torch.uint8), 1.0, n  # Return directly if all values are zero

    # Compute the sign (+1 or -1)
    sgn_x = ((x > 0).float() - 0.5) * 2

    # Calculate the quantization levels (levels = 2^n)
    levels = 2 ** n

    # Compute the probability `p`
    p = torch.abs(x) / x_norm

    # Quantize `p` to the discrete range [0, levels - 1]
    renormalize_p = p * (levels - 1)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)  # Random float for stochastic rounding
    margin = (compare < (renormalize_p - floor_p)).float()

    # Final quantized value `xi`, range is [0, (levels - 1)/(levels - 1)]
    xi = (floor_p + margin) / (levels - 1)

    # Store quantized values as integers (0 to levels - 1)
    quantized_values = (floor_p + margin).to(torch.uint8)
    print(quantized_values)

    # Pack multiple quantized values into uint8
    if n == 8:
        # 8-bit: Store each value directly, no packing required
        packed_x = quantized_values
    elif n == 4:
        # 4-bit: Pack every two values into one uint8
        high_bits = (quantized_values[::2] & 0xF) << 4  # Even indices as high 4-bit
        low_bits = quantized_values[1::2] & 0xF         # Odd indices as low 4-bit
        packed_x = high_bits | low_bits
    elif n == 2:
        # 2-bit: Pack every four values into one uint8
        packed_x = torch.zeros((quantized_values.numel() + 3) // 4, dtype=torch.uint8)
        for i in range(4):
            packed_x |= ((quantized_values[i::4] & 0x3) << (6 - 2 * i))

    return packed_x, x_norm, n, sgn_x


def decompress(packed_x, x_norm, n, sgn_x):
    """
    Decompresses the quantized tensor.

    Args:
        packed_x (torch.Tensor): Quantized tensor stored as uint8.
        x_norm (float): Scaling factor.
        n (int): Quantization bit width.
        sgn_x (torch.Tensor): Sign tensor used to restore the sign.
    Returns:
        decompressed_x (torch.Tensor): Reconstructed floating-point tensor.
    """
    if n not in [2, 4, 8]:
        raise ValueError("Only 2-bit, 4-bit, and 8-bit quantization are supported.")

    # Calculate the quantization levels (levels = 2^n)
    levels = 2 ** n

    if n == 8:
        # 8-bit: Each value is directly decompressed as a float
        quantized_values = packed_x.to(torch.float32)
    elif n == 4:
        # 4-bit: Decode every two values from one uint8
        high_bits = (packed_x >> 4).to(torch.float32)  # High 4-bit
        low_bits = (packed_x & 0xF).to(torch.float32)  # Low 4-bit
        quantized_values = torch.empty(packed_x.numel() * 2, dtype=torch.float32)
        quantized_values[::2] = high_bits
        quantized_values[1::2] = low_bits
    elif n == 2:
        # 2-bit: Decode every four values from one uint8
        quantized_values = torch.empty(packed_x.numel() * 4, dtype=torch.float32)
        for i in range(4):
            quantized_values[i::4] = ((packed_x >> (6 - 2 * i)) & 0x3).to(torch.float32)

    # Restore quantized values to the range [0, 1], then dequantize to floats
    decompressed_x = (quantized_values / (levels - 1)) * x_norm

    # Restore the sign
    decompressed_x = decompressed_x * sgn_x

    return decompressed_x


# Example data
x = torch.tensor([1.23, -3.45, 6.78, -0.12, 4.56, -7.89, 0.0, 2.34], dtype=torch.float32)

# Quantize to 4-bit
packed_x, x_norm, n, sgn_x = quantize(x, {'n': 4})
print("Quantized data (uint8):", packed_x)
print("Scaling factor:", x_norm)
print("Sign tensor:", sgn_x)

# Decompress
decompressed_x = decompress(packed_x, x_norm, n, sgn_x)
print("Decompressed data:", decompressed_x)

# Compute error
error = torch.norm(x - decompressed_x) / torch.norm(x)
print("Relative error:", error.item())