# True Implementation of Low-Precision Quantization with Bit Packing

I have reviewed a large amount of code related to quantization. Since PyTorch does not support 4-bit integers, most methods involving low-precision quantization (e.g., 2-bit and 4-bit) rely on **simulated quantization**, where the storage format remains `FP32` or `int8`.

To address this, I explored the idea of **Bit Packing**. The process is as follows:

- **Quantization**:  
  After quantizing data into 4-bit values, two 4-bit values are packed into a single `int8`.

- **Dequantization**:  
  The `int8` value is decoded to retrieve the two 4-bit values, which are then restored to their corresponding floating-point values.

## Example Implementation

I have provided an example of a **true implementation** of random uniform quantization, along with two test codes to demonstrate it.
