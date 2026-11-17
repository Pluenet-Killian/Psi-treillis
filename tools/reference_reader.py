#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Second reader of the .psi files of Psi-treillis, in Python 3 and NumPy, without a GPU.
#
# Written from docs/FORMAT.md alone, without reading src/ or the first reference reader: it serves to say whether the
# specification is sufficient on its own (claim REF-2, log benchmarks/independent_readers/python/). The "section"
# references of the comments designate the sections of docs/FORMAT.md.
#
#   python tools/reference_reader.py f.psi
#       reads back all the frames, compares the fingerprints with the table
#   python tools/reference_reader.py f.psi --probe f.probe
#       also compares lattice, fine level and field with the probe
#   python tools/reference_reader.py f.psi --frame 3 --points p.txt
#       velocity field of frame 3 at the points of p.txt
#   python tools/reference_reader.py f.psi --header
#       headers, table and seal, without decoding
#
# Exit code: 0 if everything conforms; 1 if the file is refused or if a fingerprint differs; 2 for an incorrect usage.
# Reads PSISTRM2 and PSISTRM3, with or without a seal. PSISTRM1 is refused: this reader does not support it.
#
# As a module:
#   import reference_reader
#   f = reference_reader.PsiFile("f.psi")       # raises reference_reader.Refusal if the file is invalid
#   for t in f.frames():                        # sequential decoding
#       t.fingerprint == t.table_fingerprint
#   v = f.velocity(t, points)                   # points: array (P, 3) in the box of side 2 pi; returns (P, 3)

from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

SIGNATURES = (b"PSISTRM1", b"PSISTRM2", b"PSISTRM3")
CLIP_MARKER = 0x3150494C43495350    # "PSICLIP1", section 2
SEAL_SIGNATURE = b"PSISEAL1"        # section 4.1
PROBE_SIGNATURE = b"PSIPROBE"       # section 11
EX_FINE = 0x80000000                # bit 31 of an exception: overflow of level 0, section 6.3
BLOCK = 1024                        # symbols per rANS block, section 6.1
RANS_LOW = 1 << 23                  # lower bound of the rANS state, section 6.1
READ_MARGIN = 2056                  # zero bytes added after a frame: a block reads 2052 bytes at most, section 6.1
SIZE_HEADER, SIZE_FINE, SIZE_FRAME, SIZE_SEAL = 208, 64, 48, 48
TWO_PI = 2.0 * math.pi


class Refusal(Exception):
    """Invalid file or frame: the reader refuses to read."""


# --- Fingerprint (section 7) -----------------------------------------------------------------------------------------

_M1, _M2, _M3 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def fingerprint(data, salt: int) -> int:
    """Exclusive OR, over all the positions i, of mix64(salt + 256 * i + a[i]), modulo 2^64. a[i] is read
    unsigned."""
    a = np.frombuffer(bytes(data), dtype=np.uint8) if not isinstance(data, np.ndarray) else data.reshape(-1).view(np.uint8)
    total = np.uint64(0)
    step = 1 << 22
    for start in range(0, a.size, step):
        chunk = a[start:start + step]
        x = np.arange(start, start + chunk.size, dtype=np.uint64) * np.uint64(256) + chunk.astype(np.uint64)
        x += np.uint64(salt & 0xFFFFFFFFFFFFFFFF)
        x += _M1
        x = (x ^ (x >> np.uint64(30))) * _M2
        x = (x ^ (x >> np.uint64(27))) * _M3
        x ^= x >> np.uint64(31)
        total ^= np.bitwise_xor.reduce(x)
    return int(total)


# --- Structures (sections 1 to 4) ------------------------------------------------------------------------------------

@dataclass
class Header:
    magic: bytes
    n: int
    degree: int
    mf: int
    levels: int
    fb: int
    key_interval: int
    n_frames: int
    first_frame: int
    delta: float
    dt_frame: float
    u_rms: float
    re_lambda: float
    kmax_eta: float
    eps: float
    coef: list            # coef[l] = (a, b, c), in 1/1024
    index_offset: int
    reserved: tuple       # eight words of 64 bits

    @property
    def marker(self) -> bool:
        return self.reserved[0] == CLIP_MARKER

    @property
    def clippings(self) -> tuple:
        return tuple(self.reserved[1:7])

    @property
    def clipped_frames(self) -> int:
        return self.reserved[7] & 0xFFFFFFFF

    @property
    def wrapped_frames(self) -> int:
        return self.reserved[7] >> 32


@dataclass
class FineInfo:
    tolerance: float
    eps: float
    delta: float
    max_error: float
    max_bricks: int
    max_nodes: int
    brick: int
    frames_over: int
    reserved: tuple

    @property
    def margin(self) -> float:
        """Arithmetic margin: the bits of a float32 in reserved[0], as a fraction of u' (section 3)."""
        return struct.unpack("<f", struct.pack("<I", self.reserved[0]))[0]


@dataclass
class FrameRecord:
    offset: int
    bytes: int
    hash: int
    type: int
    mean: tuple
    field_err: float
    field_max: float


@dataclass
class FineLevel:
    """Sparse fine level of a frame: the three decoded byte arrays of its section (section 6.6)."""
    n_bricks: int
    n_nodes: int
    brick_map: np.ndarray  # bricks / 8 bytes
    masks: np.ndarray      # 8 * n_bricks bytes
    coefs: np.ndarray      # 3 * n_nodes bytes


@dataclass
class Frame:
    position: int
    type: int
    states: list                      # per level: integers (nl, nl, nl, 3), indices [z, y, x, component]
    overflows: tuple                  # (positions in level 0, values): exceptions whose bit 31 is 1 (section 6.3)
    fine: Optional[FineLevel]
    mean: tuple
    fingerprint: int
    table_fingerprint: int
    n_exceptions: int = 0
    n_escapes: int = 0
    notes: list = field(default_factory=list)

    @property
    def conforming(self) -> bool:
        return self.fingerprint == self.table_fingerprint


class Geometry:
    """Quantities derived from the header (section 5)."""

    def __init__(self, e: Header):
        self.L = e.levels
        self.nl = [(e.n // e.mf) >> l for l in range(e.levels)]
        self.nodes = [m ** 3 for m in self.nl]
        self.ecount = 3 * sum(self.nodes)
        self.seg = [0] * e.levels                  # the coarsest level comes first
        for l in range(e.levels - 2, -1, -1):
            self.seg[l] = self.seg[l + 1] + 3 * self.nodes[l + 1]
        self.n_blocks = -(-self.ecount // BLOCK)
        self.key_bytes = 3 * self.nodes[0] + 6 * sum(self.nodes[1:])
        self.kcount = self.key_bytes
        self.n_kblocks = -(-self.kcount // BLOCK)
        self.ex_cap = max(65536, self.ecount // 64)


def _power_of_two(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


# --- rANS decoding (section 6.1) -------------------------------------------------------------------------------------

def decode_rans(buffer: np.ndarray, size: int, start: int, zone_end: int, limits: list, what: str) -> np.ndarray:
    """Decodes an rANS section: tables, offsets, streams.

    buffer   : bytes of the frame, followed by at least READ_MARGIN zero bytes (a block of an altered file can read
               after the end of its frame; reading zeros there is a conforming reading, section 10 point 8).
    size     : size of the frame, without the margin.
    start    : position of the first frequency table in the frame.
    zone_end : position of the end of the stream area.
    limits   : end position of each segment, in symbols; the last one is the number of symbols.
    All the blocks advance together, symbol by symbol.
    """
    n_tables, total = len(limits), int(limits[-1])
    n_blocks = -(-total // BLOCK)
    tables_end = start + 512 * n_tables
    streams_start = tables_end + 4 * n_blocks
    if start < 0 or streams_start > zone_end or zone_end > size:
        raise Refusal(f"{what}: tables and offsets outside the frame")
    freq = buffer[start:tables_end].view("<u2").reshape(n_tables, 256).astype(np.int64)
    sums = freq.sum(axis=1)
    if np.any(sums != 4096):
        t = int(np.flatnonzero(sums != 4096)[0])
        raise Refusal(f"{what}: frequency table {t} has sum {int(sums[t])}, 4096 expected")
    cum = np.cumsum(freq, axis=1) - freq                       # cum[v]: sum of the frequencies of the lower values
    sym_table = np.repeat(np.tile(np.arange(256, dtype=np.int64), n_tables), freq.reshape(-1))   # (n_tables * 4096,)
    flat_freq, flat_cum = freq.reshape(-1), cum.reshape(-1)

    zone = zone_end - streams_start
    offsets = buffer[tables_end:streams_start].view("<u4").astype(np.int64)
    if offsets[0] != 0:
        raise Refusal(f"{what}: the first block offset is not zero")
    if n_blocks > 1 and np.any(np.diff(offsets) < 4):
        raise Refusal(f"{what}: block offsets not increasing by at least 4 bytes")
    if offsets[-1] + 4 > zone:
        raise Refusal(f"{what}: the last block starts less than 4 bytes from the end of the stream area")

    p = streams_start + offsets
    x = (buffer[p].astype(np.int64) | (buffer[p + 1].astype(np.int64) << 8)
         | (buffer[p + 2].astype(np.int64) << 16) | (buffer[p + 3].astype(np.int64) << 24))
    x = np.maximum(x, RANS_LOW)                                # altered file only
    p = p + 4
    base = np.arange(n_blocks, dtype=np.int64) * BLOCK
    lim = np.asarray(limits, dtype=np.int64)
    last = total - (n_blocks - 1) * BLOCK                      # length of the last block
    out = np.empty((BLOCK, n_blocks), dtype=np.uint8)
    for k in range(BLOCK):
        m = n_blocks if k < last else n_blocks - 1             # blocks still active
        if m == 0:
            break
        xk = x[:m]
        t = np.searchsorted(lim, base[:m] + k, side="right")   # table of the segment where this symbol falls
        s = xk & 4095
        sym = sym_table[t * 4096 + s]
        j = t * 256 + sym
        xk = flat_freq[j] * (xk >> 12) + s - flat_cum[j]
        for _ in range(2):                                     # two bytes at most per symbol, section 6.1
            low = np.flatnonzero(xk < RANS_LOW)
            if low.size == 0:
                break
            xk[low] = (xk[low] << 8) | buffer[p[low]]
            p[low] += 1
        x[:m] = xk
        out[k, :m] = sym
    return np.ascontiguousarray(out.T).reshape(-1)[:total]


# --- File ------------------------------------------------------------------------------------------------------------

class PsiFile:
    """An open .psi file: headers, table and seal checked; frames decoded on demand."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            self._bytes = f.read()
        self.size = len(self._bytes)
        self._open()
        self._slots: list = [None, None]         # states of the last two decoded frames, section "Vocabulary", "slot"
        self._slot_position: list = [None, None]
        self.overflows_32_bits = 0               # predictions whose sum leaves 32 bits (should not happen)

    # -- Opening: sections 1 to 4, limits of section 9 --

    def _open(self) -> None:
        o = self._bytes
        if len(o) < SIZE_HEADER:
            raise Refusal(f"file of {len(o)} bytes: shorter than a header of {SIZE_HEADER} bytes")
        c = struct.unpack_from("<8s8i6f18iQ8Q", o, 0)
        e = Header(c[0], *c[1:9], *c[9:15], [tuple(c[15 + 3 * l:18 + 3 * l]) for l in range(6)], c[33], tuple(c[34:42]))
        self.header = e
        if e.magic not in SIGNATURES:
            raise Refusal("unknown signature: this is not a .psi file")
        if e.magic == b"PSISTRM1":
            raise Refusal("PSISTRM1: version not supported by this reader")
        self.v3 = e.magic == b"PSISTRM3"
        if e.degree not in (2, 3):
            raise Refusal(f"degree = {e.degree}: 2 or 3 expected")
        if not (_power_of_two(e.n) and 16 <= e.n <= 8192):
            raise Refusal(f"n = {e.n}: a power of two from 16 to 8192 is expected")
        if not 2 <= e.n_frames <= 1 << 24:
            raise Refusal(f"n_frames = {e.n_frames}: from 2 to 2^24 expected")
        if e.index_offset < SIZE_HEADER:
            raise Refusal(f"index_offset = {e.index_offset}: interrupted encoding or altered header")
        if not 1 <= e.levels <= 6:
            raise Refusal(f"levels = {e.levels}: from 1 to 6 expected")
        if not _power_of_two(e.mf):
            raise Refusal(f"mf = {e.mf}: a power of two is expected")
        if not 0 <= e.fb <= 12:
            raise Refusal(f"fb = {e.fb}: from 0 to 12 expected")
        g = Geometry(e)
        self.geo = g
        if min(g.nl) < 8:
            raise Refusal(f"levels of {g.nl} nodes per axis: at least 8 expected at each level")
        if g.ecount >= 1 << 31:
            raise Refusal(f"ecount = {g.ecount}: must stay under 2^31")
        if (e.levels - 1) * (32768 << e.fb) >= (1 << 25 if e.degree == 2 else 1 << 22):
            raise Refusal(f"levels = {e.levels} and fb = {e.fb}: the integer refinement would overflow in degree {e.degree}")

        self.headers_end = SIZE_HEADER + (SIZE_FINE if self.v3 else 0)
        if e.index_offset < self.headers_end:
            raise Refusal(f"index_offset = {e.index_offset}: the table starts before the end of the headers")
        table_end = e.index_offset + SIZE_FRAME * e.n_frames
        if self.size == table_end:
            self.seal = "absent"
        elif self.size == table_end + SIZE_SEAL:
            self.seal = "present"
        else:
            raise Refusal(f"size of {self.size} bytes: {table_end} expected without a seal, {table_end + SIZE_SEAL} with one")

        self.fine_info: Optional[FineInfo] = None
        if self.v3:
            c = struct.unpack_from("<4f2IiI8I", o, SIZE_HEADER)
            self.fine_info = FineInfo(*c[0:8], tuple(c[8:16]))
            if self.fine_info.brick != 4:
                raise Refusal(f"fine level: brick = {self.fine_info.brick}, 4 expected")
            if not 32 <= e.n <= 1024:
                raise Refusal(f"fine level: n = {e.n}, from 32 to 1024 expected")
            self.bricks = (e.n // 4) ** 3

        if self.seal == "present":
            magic, h_header, h_fine, h_table, r0, r1 = struct.unpack_from("<8s3Q2Q", o, table_end)
            if magic != SEAL_SIGNATURE or r0 or r1:
                raise Refusal("the 48 bytes that follow the table are not a seal")
            if h_header != fingerprint(o[:SIZE_HEADER], 0x3000):
                raise Refusal("seal: the header has changed")
            if h_fine != (fingerprint(o[SIZE_HEADER:SIZE_HEADER + SIZE_FINE], 0x3000 + (1 << 56)) if self.v3 else 0):
                raise Refusal("seal: the header of the fine level has changed")
            if h_table != fingerprint(o[e.index_offset:table_end], 0x3000 + (2 << 56)):
                raise Refusal("seal: the frame table has changed")
            self.seal = "verified"

        # Worst case of the coder, section 10 point 6.
        worst = max(g.key_bytes, g.L * 512 + 2060 * g.n_blocks) + 8 + 8 * g.ex_cap
        if self.v3:
            f = self.fine_info
            nblk = -(-(self.bricks // 8 + 8 * f.max_bricks + 3 * f.max_nodes) // BLOCK)
            worst += 8 + 3 * 512 + 2060 * nblk + 8 + 16
        self.worst_case = worst

        self.table: list = []
        for j in range(e.n_frames):
            c = struct.unpack_from("<QQQi3fff", o, e.index_offset + SIZE_FRAME * j)
            r = FrameRecord(c[0], c[1], c[2], c[3], tuple(c[4:7]), c[7], c[8])
            if not 0 <= r.type <= 3:
                raise Refusal(f"frame {j}: type {r.type}, from 0 to 3 expected")
            if r.offset < self.headers_end or r.offset + r.bytes > e.index_offset:
                raise Refusal(f"frame {j}: payload outside the frame area")
            if r.bytes > worst:
                raise Refusal(f"frame {j}: {r.bytes} bytes, more than the worst case of the coder ({worst})")
            self.table.append(r)
        if self.table[0].type not in (0, 3):
            raise Refusal("frame 0 is not a key frame")

        # Reading of the sampling lattice (section 6.5): direct if the marker is there and no frame is wrapped.
        self.direct_read = e.marker and e.wrapped_frames == 0

    # -- Payload of a frame: sections 5 and 6 --

    def _exceptions(self, payload: np.ndarray, pos: int, base_end: int, j: int):
        """Exception list (section 6.3). Returns (idx, val, end position)."""
        if pos + 8 > base_end:
            raise Refusal(f"frame {j}: exception list outside the frame")
        n_ex = int(payload[pos:pos + 4].view("<u4")[0])
        if n_ex > self.geo.ex_cap:
            raise Refusal(f"frame {j}: {n_ex} exceptions, more than ex_cap = {self.geo.ex_cap}")
        end = pos + 8 + 8 * n_ex
        if end > base_end:
            raise Refusal(f"frame {j}: exception list outside the frame")
        entries = payload[pos + 8:end].view([("idx", "<u4"), ("val", "<i4")])
        return entries["idx"].astype(np.int64), entries["val"].astype(np.int64), end

    def _fine_section(self, payload: np.ndarray, buffer: np.ndarray, j: int):
        """Section of the fine level, found from the end of the frame (section 5). Returns (FineLevel, start position
        of the section)."""
        size = payload.size
        if size % 8 or size < 16:
            raise Refusal(f"frame {j}: payload of {size} bytes, a multiple of 8 of at least 16 bytes expected")
        n_bricks, n_nodes, section_bytes, _ = struct.unpack_from("<4I", payload, size - 16)
        f = self.fine_info
        if n_bricks > f.max_bricks or n_nodes > f.max_nodes:
            raise Refusal(f"frame {j}: fine level of {n_bricks} bricks and {n_nodes} nodes, above the maxima of the header")
        if section_bytes % 8 or section_bytes > size - 16:
            raise Refusal(f"frame {j}: section of the fine level of {section_bytes} bytes, outside the frame or not a multiple of 8")
        start = size - 16 - section_bytes
        a, b = self.bricks // 8, self.bricks // 8 + 8 * n_bricks
        sym = decode_rans(buffer, size, start, size - 16, [a, b, b + 3 * n_nodes], f"frame {j}, fine level")
        return FineLevel(n_bricks, n_nodes, sym[:a], sym[a:b], sym[b:]), start

    def _decode(self, j: int) -> Frame:
        g, e, r = self.geo, self.header, self.table[j]
        payload = np.frombuffer(self._bytes, dtype=np.uint8, count=r.bytes, offset=r.offset)
        buffer = np.zeros(r.bytes + READ_MARGIN, dtype=np.uint8)
        buffer[:r.bytes] = payload
        fine, base_end = None, r.bytes
        if self.v3:                                # FineTrailer first: it bounds the base part (section 5)
            fine, base_end = self._fine_section(payload, buffer, j)
        notes: list = []

        if r.type == 0:                            # raw key
            if g.key_bytes + 8 > base_end:
                raise Refusal(f"frame {j}: raw key shorter than key_bytes = {g.key_bytes}")
            idx, val, list_end = self._exceptions(payload, g.key_bytes, base_end, j)
            expected = -(-list_end // 8) * 8 if self.v3 else list_end
            if expected != base_end:
                raise Refusal(f"frame {j}: raw key of {base_end} bytes, {expected} expected")
            states, pos = [], 0
            states.append(payload[:3 * g.nodes[0]].view(np.int8).astype(np.int32).reshape(g.nl[0], g.nl[0], g.nl[0], 3))
            pos = 3 * g.nodes[0]
            for l in range(1, g.L):
                m = 6 * g.nodes[l]
                states.append(payload[pos:pos + m].view("<i2").astype(np.int32).reshape(g.nl[l], g.nl[l], g.nl[l], 3))
                pos += m
        elif r.type == 3:                          # coded key: Lorenzo residual (section 6.4)
            idx, val, list_end = self._exceptions(payload, 0, base_end, j)
            limits, running = [], 0
            for l in range(g.L - 1, 0, -1):        # high-order bytes of level l, then its low-order bytes
                running += 3 * g.nodes[l]
                limits.append(running)
                running += 3 * g.nodes[l]
                limits.append(running)
            limits.append(running + 3 * g.nodes[0])
            sym = decode_rans(buffer, r.bytes, list_end, base_end, limits, f"frame {j}, coded key")
            states, pos = [None] * g.L, 0
            for l in range(g.L - 1, 0, -1):
                m = 3 * g.nodes[l]
                z = sym[pos + m:pos + 2 * m].astype(np.int64) | (sym[pos:pos + m].astype(np.int64) << 8)
                pos += 2 * m
                res = ((z >> 1) ^ -(z & 1)) & 0xFFFF                       # unfolding, modulo 2^16
                q = res.reshape(g.nl[l], g.nl[l], g.nl[l], 3)
                for axis in (2, 1, 0):                                      # cumulative sums along x, y, z
                    q = np.cumsum(q, axis=axis) & 0xFFFF
                states[l] = q.astype(np.uint16).view(np.int16).astype(np.int32)
            q = sym[pos:pos + 3 * g.nodes[0]].astype(np.int64).reshape(g.nl[0], g.nl[0], g.nl[0], 3)
            for axis in (2, 1, 0):
                q = np.cumsum(q, axis=axis) & 0xFF
            states[0] = q.astype(np.uint8).view(np.int8).astype(np.int32)
        else:                                      # innovation (section 6.2)
            idx, val, list_end = self._exceptions(payload, 0, base_end, j)
            limits = [g.seg[l] + 3 * g.nodes[l] for l in range(g.L - 1, -1, -1)]   # from the coarsest to the finest
            sym = decode_rans(buffer, r.bytes, list_end, base_end, limits, f"frame {j}, innovations")
            innov = sym.view(np.int8)
            if self._slot_position[(j - 1) & 1] != j - 1:
                raise Refusal(f"frame {j}: innovation without a decoded previous frame")
            q1 = self._slots[(j - 1) & 1]          # previous frame
            q2 = self._slots[j & 1]                # what this slot contained: the frame before it in sequential reading
            states = []
            for l in range(g.L):
                a, b, c = e.coef[l] if r.type == 1 else (1024, 0, 0)
                k = q1[l].astype(np.int64)
                total = a * k + 512
                bound = abs(a) * 32767 + 512
                if c:
                    lap = -6 * k
                    for axis in (0, 1, 2):
                        lap += np.roll(k, 1, axis=axis) + np.roll(k, -1, axis=axis)
                    total += c * lap
                    bound += abs(c) * 12 * 32767
                if b:
                    if q2 is None:
                        raise Refusal(f"frame {j}: innovation of type 1 without a frame k-1 (decoding not defined, section 10 point 7)")
                    total += b * q2[l].astype(np.int64)
                    bound += abs(b) * 32767
                if bound >= 1 << 31 and (total.max() >= 1 << 31 or total.min() < -(1 << 31)):
                    self.overflows_32_bits += 1
                    notes.append(f"level {l}: the sum of the prediction leaves 32 bits")
                state = (total >> 10) + innov[g.seg[l]:g.seg[l] + 3 * g.nodes[l]].reshape(k.shape)
                limit = 127 if l == 0 else 32767
                states.append(np.clip(state, -limit, limit).astype(np.int32))

        # Exceptions: escapes first (first entries whose bit 31 is zero), overflows afterwards (section 6.3).
        high = (idx & EX_FINE) != 0
        n_esc = int(idx.size if not high.any() else np.argmax(high))
        if r.type in (1, 2) and n_esc:             # a key frame applies no escape
            ei, ev = idx[:n_esc], val[:n_esc]
            for l in range(g.L):
                here = (ei >= g.seg[l]) & (ei < g.seg[l] + 3 * g.nodes[l])
                if here.any():
                    flat = states[l].reshape(-1)
                    flat[ei[here] - g.seg[l]] = ev[here].astype(np.int8 if l == 0 else np.int16)
            outside = int(np.count_nonzero(ei >= g.ecount))
            if outside:
                notes.append(f"{outside} escape(s) whose position is outside the frame, ignored")
        di = (idx[n_esc:][high[n_esc:]] & 0x7FFFFFFF) - g.seg[0]
        dv = val[n_esc:][high[n_esc:]]
        kept = (di >= 0) & (di < 3 * g.nodes[0])   # an overflow outside level 0 is ignored
        overflows = (di[kept], dv[kept])

        h = fingerprint(states[0].astype(np.int8), 0x1000)
        for l in range(1, g.L):
            h ^= fingerprint(states[l].astype("<i2"), 0x1000 + (l << 56))
        if fine is not None:
            h ^= fingerprint(fine.brick_map, 0x2000) ^ fingerprint(fine.masks, 0x2000 + (1 << 56)) ^ fingerprint(fine.coefs, 0x2000 + (2 << 56))

        self._slots[j & 1], self._slot_position[j & 1] = states, j
        return Frame(j, r.type, states, overflows, fine, r.mean, h, r.hash, int(idx.size), n_esc, notes)

    def frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[Frame]:
        """Decodes the frames from `start` to `end` (exclusive), in order. `start` must be a key frame."""
        if self.table[start].type not in (0, 3):
            raise Refusal(f"frame {start} is not a key frame: the decoding cannot start there")
        self._slots, self._slot_position = [None, None], [None, None]
        for j in range(start, self.header.n_frames if end is None else end):
            yield self._decode(j)

    def frame(self, j: int) -> Frame:
        """Decodes frame j, starting again from the key frame that precedes it."""
        if not 0 <= j < self.header.n_frames:
            raise Refusal(f"frame {j}: the file has {self.header.n_frames} frames")
        key = j
        while self.table[key].type not in (0, 3) or (key > 0 and key + 1 <= j and self.table[key + 1].type == 1):
            key -= 1                               # an innovation of type 1 just after a key needs the frame before it
        t = None
        for t in self.frames(key, j + 1):
            pass
        return t

    # -- Sampling lattice: section 6.5 --

    def _refine(self, g: np.ndarray) -> np.ndarray:
        """Refinement of a sum of m nodes per axis to 2 m nodes per axis: integer masks, a single division."""
        for axis in (0, 1, 2):
            shape = list(g.shape)
            shape[axis] *= 2
            fine = np.empty(shape, dtype=np.int64)
            even = [slice(None)] * 4
            odd = [slice(None)] * 4
            even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
            minus, plus = np.roll(g, 1, axis=axis), np.roll(g, -1, axis=axis)   # g[i-1], g[i+1]
            if self.header.degree == 2:
                fine[tuple(even)] = 3 * g + minus
                fine[tuple(odd)] = 3 * g + plus
            else:
                fine[tuple(even)] = minus + 6 * g + plus
                fine[tuple(odd)] = 4 * g + 4 * plus
            g = fine
        return (g + 32) >> 6 if self.header.degree == 2 else (g + 256) >> 9

    def lattice(self, t: Frame) -> np.ndarray:
        """Sampling lattice of the frame, modulo 2^16: int16 (nl0, nl0, nl0, 3), overflows included."""
        fb, L = self.header.fb, self.geo.L
        total = t.states[L - 1].astype(np.int64) << fb
        for l in range(L - 2, -1, -1):
            total = (t.states[l].astype(np.int64) << fb) + self._refine(total)
        positions, values = t.overflows
        if positions.size:
            np.add.at(total.reshape(-1), positions, values << fb)
        return (total & 0xFFFF).astype(np.uint16).view(np.int16)

    # -- Sparse fine level: section 6.6 --

    def dense_fine_level(self, t: Frame) -> Optional[np.ndarray]:
        """Coefficients of the fine level on the grid of step one voxel: int8 (n, n, n, 3), zero for a node that is not
        retained."""
        if t.fine is None:
            return None
        n, f = self.header.n, t.fine
        nb = n // 4
        dense = np.zeros((n, n, n, 3), dtype=np.int8)
        present = np.flatnonzero(np.unpackbits(f.brick_map, bitorder="little"))[:f.n_bricks]   # what exceeds: absent
        if present.size == 0:
            return dense
        bits = np.unpackbits(f.masks[:8 * present.size].reshape(-1, 8), axis=1, bitorder="little")
        brick, k = np.nonzero(bits)                # in the order of the bricks, then of the increasing bits k
        brick, k = brick[:f.n_nodes], k[:f.n_nodes]
        b = present[brick]
        x = 4 * (b % nb) + (k & 3)
        y = 4 * ((b // nb) % nb) + ((k >> 2) & 3)
        z = 4 * (b // (nb * nb)) + (k >> 4)
        dense[z, y, x, :] = f.coefs.view(np.int8).reshape(-1, 3)[:brick.size]
        return dense

    # -- Velocity field: section 6.7 --

    @property
    def scale(self) -> float:
        """E = delta / (2^fb * h0), with h0 = 2 pi / nl[0]."""
        return self.header.delta / ((1 << self.header.fb) * (TWO_PI / self.geo.nl[0]))

    @property
    def fine_scale(self) -> float:
        """Ef = FineInfo.delta * n / 2 pi; zero without a fine level."""
        return self.fine_info.delta * self.header.n / TWO_PI if self.v3 else 0.0

    def velocity(self, t: Frame, points, lattice: Optional[np.ndarray] = None, dense: Optional[np.ndarray] = None,
                 without_fine_level: bool = False) -> np.ndarray:
        """Velocity field of the frame at points (P, 3) of the periodic box of side 2 pi, in double precision."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if lattice is None:
            lattice = self.lattice(t)
        d = self.header.degree
        v = self.scale * curl(lattice, pts, TWO_PI / self.geo.nl[0], d, not self.direct_read)
        if self.v3 and not without_fine_level:
            if dense is None:
                dense = self.dense_fine_level(t)
            v += self.fine_scale * curl(dense, pts, TWO_PI / self.header.n, d, False)
        return v + np.asarray(t.mean, dtype=np.float64)


def _weights(u: np.ndarray, degree: int):
    """First node of the support, weights and derivatives of the cardinal B-spline along one axis (section 6.7)."""
    if degree == 2:
        m = np.floor(u + 0.5)
        r = u - m
        w = [(0.5 - r) ** 2 / 2, 0.75 - r * r, (0.5 + r) ** 2 / 2]
        dw = [-(0.5 - r), -2 * r, 0.5 + r]
        return m.astype(np.int64) - 1, w, dw
    i = np.floor(u)
    t = u - i
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    dw = [-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2, (-3 * t ** 2 + 2 * t + 1) / 2, t ** 2 / 2]
    return i.astype(np.int64) - 1, w, dw


def curl(T: np.ndarray, pts: np.ndarray, step: float, degree: int, differences: bool) -> np.ndarray:
    """Curl, with respect to u = x / step - d, of the potential of integer coefficients T (N, N, N, 3) at the points
    pts.

    differences : reading by differences (section 6.5): the value of the first node of the support is subtracted from
    each node, modulo 2^16, and the difference is read back as a signed 16-bit integer.
    """
    N = T.shape[0]
    d = 0.5 if degree == 2 else 0.0
    ix, wx, dwx = _weights(pts[:, 0] / step - d, degree)
    iy, wy, dwy = _weights(pts[:, 1] / step - d, degree)
    iz, wz, dwz = _weights(pts[:, 2] / step - d, degree)
    rot = np.zeros((pts.shape[0], 3), dtype=np.float64)
    first = T[iz % N, iy % N, ix % N].astype(np.int64) if differences else None
    for kz in range(degree + 1):
        for ky in range(degree + 1):
            for kx in range(degree + 1):
                q = T[(iz + kz) % N, (iy + ky) % N, (ix + kx) % N].astype(np.int64)
                if differences:
                    q = ((q - first + 32768) & 0xFFFF) - 32768
                gx = dwx[kx] * wy[ky] * wz[kz]
                gy = wx[kx] * dwy[ky] * wz[kz]
                gz = wx[kx] * wy[ky] * dwz[kz]
                rot[:, 0] += gy * q[:, 2] - gz * q[:, 1]       # dPsi_z/dy - dPsi_y/dz
                rot[:, 1] += gz * q[:, 0] - gx * q[:, 2]       # dPsi_x/dz - dPsi_z/dx
                rot[:, 2] += gx * q[:, 1] - gy * q[:, 0]       # dPsi_y/dx - dPsi_x/dy
    return rot


# --- Probe file (section 11) -----------------------------------------------------------------------------------------

@dataclass
class Probe:
    n: int
    degree: int
    nl0: int
    frame: int            # counted in the source sequence
    points: int
    base_points: int
    fine_present: int
    wrapped: int
    scale: float
    fine_scale: float
    eighth: float
    u_rms: float
    threshold: float
    mean: tuple
    lattice: np.ndarray               # int16 (nl0, nl0, nl0, 3)
    fine: Optional[np.ndarray]        # int8 (n, n, n, 3)
    positions: np.ndarray             # float64 (P, 3)
    velocities: np.ndarray            # float64 (P, 3)


def read_probe(path: str) -> Probe:
    with open(path, "rb") as f:
        o = f.read()
    if len(o) < 72 or o[:8] != PROBE_SIGNATURE:
        raise Refusal("this is not a probe file")
    n, degree, nl0, frame, P, base, fine_present, wrapped = struct.unpack_from("<8I", o, 8)
    E, Ef, eighth, u_rms, threshold, mx, my, mz = struct.unpack_from("<8f", o, 40)
    expected = 72 + 6 * nl0 ** 3 + (3 * n ** 3 if fine_present else 0) + 24 * P
    if len(o) != expected:
        raise Refusal(f"probe of {len(o)} bytes, {expected} expected")
    pos = 72
    lattice = np.frombuffer(o, "<i2", 3 * nl0 ** 3, pos).reshape(nl0, nl0, nl0, 3)
    pos += 6 * nl0 ** 3
    fine = None
    if fine_present:
        fine = np.frombuffer(o, np.int8, 3 * n ** 3, pos).reshape(n, n, n, 3)
        pos += 3 * n ** 3
    positions = np.frombuffer(o, "<f4", 3 * P, pos).reshape(3, P).T.astype(np.float64)
    velocities = np.frombuffer(o, "<f4", 3 * P, pos + 12 * P).reshape(3, P).T.astype(np.float64)
    return Probe(n, degree, nl0, frame, P, base, fine_present, wrapped, E, Ef, eighth, u_rms, threshold, (mx, my, mz),
                 lattice, fine, positions, velocities)


@dataclass
class Comparison:
    frame: int                        # position in the file
    points: int
    differing_lattice_nodes: int
    differing_fine_nodes: int         # 0 without a fine level
    max_difference: float             # largest difference of a component, in velocity units
    max_difference_u: float           # the same, as a fraction of u'
    scale_difference: float           # relative difference between the computed E and the E of the probe
    fine_scale_difference: float


def compare_with_probe(f: PsiFile, s: Probe, t: Optional[Frame] = None) -> Comparison:
    """Compares the lattice, the fine level and the field of the probed frame with those recorded by the encoder."""
    e = f.header
    if (s.n, s.degree, s.nl0) != (e.n, e.degree, f.geo.nl[0]):
        raise Refusal("the probe is not that of this file: n, degree or nl[0] differ")
    if bool(s.fine_present) != f.v3:
        raise Refusal("the probe is not that of this file: presence of the fine level")
    position = s.frame - e.first_frame
    if not 0 <= position < e.n_frames:
        raise Refusal(f"the probe carries frame {s.frame} of the sequence, outside the file")
    if t is None or t.position != position:
        t = f.frame(position)
    lattice = f.lattice(t)
    dense = f.dense_fine_level(t)
    n_lattice = int(np.count_nonzero(np.any(lattice != s.lattice, axis=3)))
    n_fine = int(np.count_nonzero(np.any(dense != s.fine, axis=3))) if f.v3 else 0
    v = f.velocity(t, s.positions, lattice, dense)
    difference = float(np.max(np.abs(v - s.velocities)))
    return Comparison(position, s.points, n_lattice, n_fine, difference, difference / e.u_rms,
                      abs(f.scale / s.scale - 1.0) if s.scale else float("nan"),
                      abs(f.fine_scale / s.fine_scale - 1.0) if s.fine_scale else 0.0)


# --- Command line ----------------------------------------------------------------------------------------------------

def _print_header(f: PsiFile) -> None:
    e, g = f.header, f.geo
    print(f"File           : {e.magic.decode()}, {e.n}^3, degree {e.degree}, {e.levels} levels ({', '.join(str(m) for m in g.nl)} nodes per axis), "
          f"mf {e.mf}, fb {e.fb}")
    print(f"Frames         : {e.n_frames}, from frame {e.first_frame} of the sequence; one key every {e.key_interval}; "
          f"types {''.join(str(r.type) for r in f.table)}")
    print(f"Quantisation   : delta {e.delta:.6e}, eps {e.eps:g}, u' {e.u_rms:.6f}, dt {e.dt_frame:.6g}")
    print(f"Predictors     : {'; '.join(str(e.coef[l]) for l in range(e.levels))}")
    if e.marker:
        print(f"Clippings      : sites {e.clippings}, {e.clipped_frames} clipped frame(s), {e.wrapped_frames} wrapped frame(s)")
    else:
        print("Clippings      : no marker")
    print(f"Lattice        : read {'directly' if f.direct_read else 'by differences'}")
    if f.v3:
        i = f.fine_info
        print(f"Fine level     : tolerance {i.tolerance:g} u', margin {i.margin:g}, eps {i.eps:g}, delta {i.delta:.6e}, largest error "
              f"{i.max_error:.6f} u', {i.max_bricks} bricks and {i.max_nodes} nodes at most, {i.frames_over} frame(s) above the threshold")
    print(f"Seal           : {f.seal}")


def main(arguments: list) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    usage = ("usage: python tools/reference_reader.py f.psi [--header] [--probe f.probe] [--frame j --points p.txt [--output v.txt]]\n"
             "  p.txt: one point per line, three coordinates in the box of side 2 pi")
    options = {"--probe": None, "--frame": None, "--points": None, "--output": None}
    header_only, path, i = False, None, 0
    while i < len(arguments):
        a = arguments[i]
        if a == "--header":
            header_only = True
        elif a in options and i + 1 < len(arguments):
            options[a] = arguments[i + 1]
            i += 1
        elif not a.startswith("--") and path is None:
            path = a
        else:
            print(usage, file=sys.stderr)
            return 2
        i += 1
    if path is None or (options["--points"] is None) != (options["--frame"] is None):
        print(usage, file=sys.stderr)
        return 2
    try:
        f = PsiFile(path)
        if options["--points"] is not None:        # the velocities alone, one line per point
            t = f.frame(int(options["--frame"]))
            v = f.velocity(t, np.loadtxt(options["--points"], dtype=np.float64, ndmin=2))
            np.savetxt(options["--output"] if options["--output"] else sys.stdout, v, fmt="%.9e")
            return 0
        _print_header(f)
        if header_only:
            return 0
        probe = read_probe(options["--probe"]) if options["--probe"] else None
        probe_position = probe.frame - f.header.first_frame if probe else None
        differing, n_read, comparison = 0, 0, None
        for t in f.frames():
            n_read += 1
            if not t.conforming:
                differing += 1
                print(f"Frame {t.position:5d}   : fingerprint {t.fingerprint:016x}, the table carries {t.table_fingerprint:016x}")
            for note in t.notes:
                print(f"Frame {t.position:5d}   : {note}")
            if probe is not None and t.position == probe_position:
                comparison = compare_with_probe(f, probe, t)
        print(f"Integrity      : {n_read} frames read back, {differing} fingerprint(s) different from that of the table")
        if probe is not None:
            if comparison is None:
                raise Refusal(f"the probe carries frame {probe.frame} of the sequence, outside the file")
            c = comparison
            print(f"Probe          : frame {c.frame}, {c.points} points; {c.differing_lattice_nodes} node(s) of the lattice and "
                  f"{c.differing_fine_nodes} node(s) of the fine level different from those of the encoder; largest difference of the field "
                  f"{c.max_difference:.3e}, that is {c.max_difference_u:.3e} u'")
        return 1 if differing else 0
    except Refusal as r:
        print(f"Refusal: {r}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as r:
        print(f"Refusal: unreadable file or argument ({type(r).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
