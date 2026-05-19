import math
from io import BytesIO
from tqdm import tqdm
import sys
from pathlib import Path
import os
# ──────────────────────────────────────────────────────────────────────────────
# Bit-level reader (mirrors StreamBitReader: reads bytes MSB-first, no padding
# between fields — exactly how HeaderCoder/ECLibDirect stores bypass symbols)
# ──────────────────────────────────────────────────────────────────────────────
class _BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._cur_byte = 0
        self._bits_left = 0   # valid bits remaining in _cur_byte

    def read(self, n: int) -> int:
        val = 0
        for _ in range(n):
            if self._bits_left == 0:
                self._cur_byte = self._data[self._byte_pos]
                self._byte_pos += 1
                self._bits_left = 8
            val = (val << 1) | ((self._cur_byte >> (self._bits_left - 1)) & 1)
            self._bits_left -= 1
        return val


# ──────────────────────────────────────────────────────────────────────────────
# Exp-Golomb k=0 size decoder (mirrors Binarizers.decode_unsigned_expgolomb_k0
# + StreamBitReader from the codebase).  The reader consumes whole bytes from f
# even when fewer bits are needed — remaining bits are discarded just like the
# temporary StreamBitReader objects in Substream.__decode_substream_size.
# ──────────────────────────────────────────────────────────────────────────────
def _read_substream_size(f) -> int:
    cur_byte = 0
    bits_left = 0

    def _bit():
        nonlocal cur_byte, bits_left
        if bits_left == 0:
            cur_byte = f.read(1)[0]
            bits_left = 8
        b = (cur_byte >> (bits_left - 1)) & 1
        bits_left -= 1
        return b

    # decode_unsigned_unary: count leading 0s until first 1
    data_size = 0
    while _bit() == 0:
        data_size += 1

    # read data_size bits for the value part
    value = 0
    for _ in range(data_size):
        value = (value << 1) | _bit()

    # Binarizers formula: ans = (1 << data_size) + (value - 1)
    return (1 << data_size) + (value - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Main parser
# Traces the exact decode_header_recursively call chain for this codec:
#
#   CodingEngine.decode_header()          → pic_header (PIH, 0xFF82)
#   ColourTransformation.decode_header()  → pic_header (child of colour_processing)
#   MultiToolsEngine.decode_header()      → pic_header (CcsGvaeMultiTools)
#
# Fields written before model_id (all bypass-coded, packed MSB-first, no gaps):
#
#   decoder_profile_id           4 bits
#   num_synthesis_transforms_minus1  4 bits  → value N
#   synthesis_transform_id[0..N] (N+1)×4 bits  (variable!)
#   level_idc                    8 bits
#   img_width_minus64            16 bits  (max=65535)
#   img_height_minus64           16 bits
#   diff_display_img_width       6 bits
#   diff_display_img_height      6 bits
#   bit_depth_idc                3 bits  (ceil(log2(5))=3, max_symbol=4)
#   s_ver_minus1                 1 bit
#   s_hor_minus1                 1 bit
#   c_ver_minus1  (if s_ver==1)  1 bit
#   c_hor_minus1  (if s_hor==1)  1 bit
#   colour_transform_idx         2 bits  (ceil(log2(3))=2, max_symbol=2)
#   colour transform matrix      96 bits only if colour_transform_idx==2
#   model_id                  ← 4 bits  (ceil(log2(16))=4, max_models_count=16)
# ──────────────────────────────────────────────────────────────────────────────
def read_model_id_raw(path: str, max_models_count: int = 16) -> int:
    MARKER_SOC = 0xFF80
    MARKER_PIH = 0xFF82
    BO = 'big'

    with open(path, 'rb') as f:
        soc = int.from_bytes(f.read(2), BO)
        assert soc == MARKER_SOC, f"Expected SOC 0xFF80, got {hex(soc)}"

        # Scan substreams until we find PIH (0xFF82)
        while True:
            marker = int.from_bytes(f.read(2), BO)
            size   = _read_substream_size(f)
            data   = f.read(size)
            if marker == MARKER_PIH:
                break

    r = _BitReader(data)

    # ── CodingEngine.decode_header ────────────────────────────────────────────
    r.read(4)                                  # decoder_profile_id
    N = r.read(4)                              # num_synthesis_transforms_minus1
    for _ in range(N + 1):
        r.read(4)                              # synthesis_transform_id[i]
    r.read(8)                                  # level_idc
    r.read(16)                                 # img_width_minus64
    r.read(16)                                 # img_height_minus64
    r.read(6)                                  # diff_display_img_width
    r.read(6)                                  # diff_display_img_height
    r.read(3)                                  # bit_depth_idc  (ceil(log2(5))=3)
    s_ver_minus1 = r.read(1)
    s_hor_minus1 = r.read(1)
    if s_ver_minus1 == 0:                      # s_ver == 1  → c_ver is signalled
        r.read(1)
    if s_hor_minus1 == 0:                      # s_hor == 1  → c_hor is signalled
        r.read(1)

    # ── ColourTransformation.decode_header ────────────────────────────────────
    colour_transform_idx = r.read(2)           # max_symbol=2, ceil(log2(3))=2 bits
    if colour_transform_idx == 2:              # custom matrix: 9×8 + 3×8 bits
        for _ in range(12):
            r.read(8)

    # ── MultiToolsEngine.decode_header (CcsGvaeMultiTools) ───────────────────
    # max_models_count=16, max_symbol=15, ceil(log2(16))=4 bits
    bits = math.ceil(math.log2(max_models_count))
    model_id = r.read(bits)
    return model_id


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_dir> <output_txt>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_txt = Path(sys.argv[2])

    bits_files = sorted(input_dir.glob("*.bits"))
    if not bits_files:
        print(f"No .bits files found in {input_dir}")
        sys.exit(1)

    with open(output_txt, "w") as f:
        for bits_file in tqdm(bits_files):
            mid = read_model_id_raw(str(bits_file))

            f.write(str(bits_file) + f",{mid}" + "\n")

    print(f"Done. Written {len(bits_files)} paths to {output_txt}")


if __name__ == "__main__":
    main()



