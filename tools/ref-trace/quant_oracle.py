"""Golden cases for the Zig weight-quantisation decoders.

The engine never re-quantises: it dequantises what ModelOpt wrote. So the
reference here is deliberately *not* produced by the Zig code, and not produced
by re-running ModelOpt either. It is an independent numpy implementation of the
two layouts that appears in the checkpoint:

* NVFP4 (the expert stacks): 4-bit E2M1 weights, two per byte (low nibble
  first), one E4M3 scale per 16 consecutive weights along the contracted
  dimension, and one FP32 second-level scale per tensor.
* FP8 E4M3 blockwise (the MTP layer): one BF16 inverse scale per 128x128 tile.

Each case stores the stored buffers, the source values, and the float32
dequantised reference. The reference is computed with the same multiply order
as the Zig decoder -- ``fp4 * (block_scale * scale_2)`` and ``fp8 * scale_inv``
-- so the comparison is exact rather than approximate.

Usage: ``python3 tools/ref-trace/quant_oracle.py`` writes the fixtures into
``tests/golden/quant/``.
"""

from __future__ import annotations

import json
import pathlib
import struct

import numpy as np

GOLDEN_DIRECTORY = pathlib.Path(__file__).resolve().parents[2] / "tests/golden/quant"

# Magnitudes E2M1 represents, indexed by the low three bits of the code.
FP4_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Values of every E2M1 code: bit 3 is the sign, and -0.0 keeps its sign.
FP4_VALUES = np.concatenate([FP4_MAGNITUDES, -FP4_MAGNITUDES]).astype(np.float32)

# The first-level NVFP4 scale saturates at the largest E2M1 magnitude.
NVFP4_SCALE_MAX_MAGNITUDE = 6.0
# The second-level scale saturates at the largest E4M3 magnitude.
FP8_E4M3_MAX_MAGNITUDE = 448.0

FP8_BLOCK_SIZE = 128


def decode_all_e4m3() -> np.ndarray:
    """Decodes every E4M3 (FN) byte code into float32, indexed by the code."""
    table = np.zeros(256, dtype=np.float32)
    for code in range(256):
        negative = bool(code & 0x80)
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            magnitude = np.float32(mantissa) * np.float32(2.0**-9)
        elif exponent == 0x0F and mantissa == 0x07:
            magnitude = np.float32(np.nan)
        else:
            magnitude = np.float32(1.0 + mantissa / 8.0) * np.float32(
                2.0 ** (exponent - 7)
            )
        table[code] = np.float32(-magnitude) if negative else magnitude
    return table


E4M3_VALUES = decode_all_e4m3()

# The sign-free half of the code space: codes 0x00..0x7E in ascending
# magnitude order. Encoding selects from these and then sets the sign bit, so a
# magnitude can never pull in a code of the opposite sign.
E4M3_POSITIVE_CODES = np.arange(0x7F, dtype=np.int64)
E4M3_MAGNITUDES = E4M3_VALUES[E4M3_POSITIVE_CODES]


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even encodes float32 into E4M3 (FN) bytes.

    E4M3 is sign-magnitude, so the magnitude is rounded on the 127 non-negative
    codes and the sign bit is applied afterwards -- the same shape as
    ``quantise_e2m1``. Encoding the signed float instead would let a tie between
    two magnitudes resolve to the *negative* code of the lower one (a positive
    420 became 0xFD, -416, before this was fixed). Ties go to the code whose
    mantissa is even, which is what a hardware conversion does. NaN and
    infinity are rejected: a scale of either is a broken checkpoint, not a
    rounding question.
    """
    flat = np.asarray(values, dtype=np.float32).ravel()
    if not np.all(np.isfinite(flat)):
        raise ValueError("cannot encode a non-finite value as E4M3")

    encoded = np.empty(flat.shape, dtype=np.uint8)
    for index, value in enumerate(flat):
        magnitude = np.float64(abs(value))
        best = _nearest_code(E4M3_POSITIVE_CODES, E4M3_MAGNITUDES, magnitude)
        if np.signbit(value):
            best |= 0x80
        encoded[index] = np.uint8(best)
    return encoded.reshape(np.shape(values))


def _nearest_code(codes: np.ndarray, magnitudes: np.ndarray, value: float) -> int:
    """Code nearest `value` in an ascending magnitude table, ties to even code.

    `codes` and `magnitudes` must be sorted by ascending magnitude and carry no
    sign bit, so the tie-break compares two mantissas of the same sign.
    """
    assert np.all(np.diff(magnitudes) > 0), "magnitude table must ascend"
    assert np.all(codes < 0x80), "codes must be sign-free"
    upper = int(np.searchsorted(magnitudes, value, side="left"))
    if upper == 0:
        return int(codes[0])
    if upper >= magnitudes.size:
        return int(codes[-1])
    low_code = int(codes[upper - 1])
    high_code = int(codes[upper])
    low_distance = abs(value - np.float64(magnitudes[upper - 1]))
    high_distance = abs(np.float64(magnitudes[upper]) - value)
    if low_distance < high_distance:
        return low_code
    if high_distance < low_distance:
        return high_code
    return low_code if low_code % 2 == 0 else high_code


def encode_bf16(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even encodes float32 into BF16 bit patterns."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    magnitude = bits & 0x7FFFFFFF
    quiet_nan = (bits >> 16) | 0x0040
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
    return np.where(magnitude > 0x7F800000, quiet_nan, rounded).astype(np.uint16)


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    """Widens BF16 bit patterns to float32; exact by construction."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def check_encoder() -> None:
    """Regression guard for the magnitude-then-sign encoding order.

    Before the fix a positive 420 selected the negative code of 416 (0xFD), so a
    NVFP4 block scale of -416 reached the fixture -- something ModelOpt cannot
    emit. The expected bytes are written out by hand on purpose.
    """
    values = np.array(
        [420.0, -420.0, 416.0, -416.0, 0.0, -0.0, 448.0, -448.0, 1e9],
        dtype=np.float32,
    )
    expected = [0x7D, 0xFD, 0x7D, 0xFD, 0x00, 0x80, 0x7E, 0xFE, 0x7E]
    got = encode_e4m3(values).tolist()
    assert got == expected, (got, expected)
    # 432 is exactly between 416 and 448: the tie goes to the even mantissa
    # (0x7E), never to the opposite-signed code of the lower magnitude.
    tie = encode_e4m3(np.array([432.0, -432.0], dtype=np.float32)).tolist()
    assert tie == [0x7E, 0xFE], tie


def quantise_e2m1(values: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even encodes float32 into the 16 E2M1 codes.

    E2M1 is sign-magnitude, so the magnitude is rounded on the eight positive
    levels and the sign bit is applied afterwards. That keeps the sign of a
    value that rounds to zero (``-0.05`` becomes the ``-0.0`` code), which is
    what a hardware conversion does and what the decoder must reproduce.
    """
    flat = np.asarray(values, dtype=np.float32).ravel()
    codes = np.empty(flat.shape, dtype=np.uint8)
    for index, value in enumerate(flat):
        best = _nearest_code(np.arange(8, dtype=np.int64), FP4_MAGNITUDES, abs(value))
        if np.signbit(value):
            best |= 0x08
        codes[index] = np.uint8(best)
    return codes.reshape(np.shape(values))


def dequantise_nvfp4(
    packed: np.ndarray, scales: np.ndarray, scale_2: np.float32
) -> np.ndarray:
    """Dequantises one NVFP4 row exactly the way the Zig decoder does."""
    weights = np.empty(packed.size * 2, dtype=np.float32)
    weights[0::2] = FP4_VALUES[packed & 0x0F]
    weights[1::2] = FP4_VALUES[packed >> 4]
    blocks = scales.size
    assert weights.size == blocks * 16, (weights.size, blocks)
    combined = np.repeat(E4M3_VALUES[scales.astype(np.uint8)] * np.float32(scale_2), 16)
    return (weights * combined).astype(np.float32)


def write_safetensors(path: pathlib.Path, tensors: dict[str, tuple]):
    """Writes a safetensors file; tensors are ``name -> (dtype, shape, bytes)``."""
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header, sort_keys=True).encode()
    encoded += b" " * ((8 - (len(encoded) % 8)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)


def float_tensor(array: np.ndarray) -> tuple:
    array = np.ascontiguousarray(array, dtype=np.float32)
    return ("F32", array.shape, array.tobytes())


def nvfp4_pattern() -> np.ndarray:
    """Two blocks of 16 values covering the E2M1 grid plus sub-step tails."""
    block_0 = np.array(
        [
            0.0, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0, 6.0,
            0.25, -0.75, 1.2, -2.9, 5.5, -6.4, 0.1, -0.05,
        ],
        dtype=np.float32,
    )
    block_1 = np.array(
        [
            6.0, 5.0, 4.5, 3.5, 2.5, 1.75, 1.25, 0.75,
            0.375, -0.375, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([block_0, block_1])


def build_nvfp4_case() -> dict[str, tuple]:
    pattern = nvfp4_pattern()
    blocks = pattern.size // 16
    assert pattern.size == blocks * 16, pattern.size

    reshaped = pattern.reshape(blocks, 16)
    block_amax = np.abs(reshaped).max(axis=1).astype(np.float32)
    global_amax = np.float32(np.abs(pattern).max())

    # NVIDIA's two-level NVFP4 scale: the second level normalises the tensor so
    # that the E4M3 block scale always fits its 448 range.
    #   scale_2 = global_amax / (6 * 448)
    #   block   = e4m3(amax_block / (6 * scale_2))
    #           = e4m3(amax_block * 448 / global_amax)
    # and w = q * (block * scale_2). Checked against the real checkpoint: the
    # reconstructed max magnitude then equals global_amax.
    scale_2 = np.float32(
        global_amax / np.float32(NVFP4_SCALE_MAX_MAGNITUDE * FP8_E4M3_MAX_MAGNITUDE)
    )
    block_scale = encode_e4m3(
        block_amax / (np.float32(NVFP4_SCALE_MAX_MAGNITUDE) * scale_2)
    )
    # A block scale is an amax; ModelOpt never writes a negative one.
    assert np.all(block_scale < 0x80), block_scale.tolist()
    block_scale_f32 = E4M3_VALUES[block_scale]

    combined = np.repeat(block_scale_f32, 16) * scale_2
    codes = quantise_e2m1(pattern / combined)

    packed = np.zeros(pattern.size // 2, dtype=np.uint8)
    packed[:] = codes[0::2] & 0x0F
    packed |= (codes[1::2] & 0x0F) << 4

    reference = dequantise_nvfp4(packed, block_scale, scale_2)
    return {
        "pattern": float_tensor(pattern),
        "packed": ("U8", packed.shape, packed.tobytes()),
        "scales": ("F8_E4M3", block_scale.shape, block_scale.tobytes()),
        "scale_2": ("F32", (), np.array(scale_2, dtype=np.float32).tobytes()),
        "reference": float_tensor(reference),
        "codes": ("U8", codes.shape, codes.tobytes()),
    }


def build_fp8_blockwise_case() -> dict[str, tuple]:
    rows = 256
    cols = 256
    positions = np.arange(rows * cols, dtype=np.float32)
    magnitudes = np.linspace(0.125, 8.0, rows * cols, dtype=np.float32)
    weight = encode_e4m3(
        (magnitudes * np.float32(0.75) * np.sin(positions)).reshape(rows, cols)
    )
    scale_inv = np.array([[0.001953125, -0.5], [3.0, 448.0]], dtype=np.float32)
    scale_inv_bits = encode_bf16(scale_inv)

    values = E4M3_VALUES[weight]
    tiled = np.repeat(
        np.repeat(decode_bf16(scale_inv_bits), FP8_BLOCK_SIZE, axis=0),
        FP8_BLOCK_SIZE,
        axis=1,
    )
    reference = (values * tiled).astype(np.float32)
    return {
        "weight": ("F8_E4M3", weight.shape, weight.tobytes()),
        "scale_inv": ("BF16", scale_inv.shape, scale_inv_bits.tobytes()),
        "reference": float_tensor(reference),
    }


def main() -> None:
    check_encoder()
    GOLDEN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    nvfp4_case = build_nvfp4_case()
    write_safetensors(GOLDEN_DIRECTORY / "nvfp4_case.safetensors", nvfp4_case)
    fp8_case = build_fp8_blockwise_case()
    write_safetensors(GOLDEN_DIRECTORY / "fp8_blockwise_case.safetensors", fp8_case)

    print(
        "nvfp4_case: scale_2=%r scales=%r codes=%r"
        % (
            float(np.frombuffer(nvfp4_case["scale_2"][2], dtype=np.float32)[0]),
            np.frombuffer(nvfp4_case["scales"][2], dtype=np.uint8).tolist(),
            np.frombuffer(nvfp4_case["codes"][2], dtype=np.uint8).tolist(),
        )
    )
    print(
        "nvfp4 reference: %r"
        % np.frombuffer(nvfp4_case["reference"][2], dtype=np.float32).tolist()
    )
    print(
        "fp8_blockwise scale_inv bits: %r"
        % np.frombuffer(fp8_case["scale_inv"][2], dtype=np.uint16).tolist()
    )
    print(
        "fp8_blockwise reference[0,:4]: %r"
        % np.frombuffer(fp8_case["reference"][2], dtype=np.float32)[:4].tolist()
    )
    print("wrote", GOLDEN_DIRECTORY)


if __name__ == "__main__":
    main()
