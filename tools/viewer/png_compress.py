# SPDX-License-Identifier: Apache-2.0
"""Recompresses in place the PNG captures of psi_view, written without compression.

Usage: python tools/viewer/png_compress.py docs/figures/*_256.png docs/figures/*_512.png
"""
import glob
import struct
import sys
import zlib


def recompress(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, chunks, idat = 8, [], b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IDAT":
            idat += body
        else:
            chunks.append((kind, body))
    packed = zlib.compress(zlib.decompress(idat), 9)
    out = bytearray(data[:8])
    for kind, body in chunks:
        if kind == b"IEND":
            out += struct.pack(">I", len(packed)) + b"IDAT" + packed + struct.pack(">I", zlib.crc32(b"IDAT" + packed))
        out += struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    open(path, "wb").write(out)
    return len(data), len(out)


if __name__ == "__main__":
    for pattern in sys.argv[1:]:
        for path in glob.glob(pattern):
            before, after = recompress(path)
            print(f"{path}: {before} -> {after} bytes")
