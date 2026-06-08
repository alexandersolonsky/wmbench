"""Shared BCH error-correction helper (via ``bchlib``).

Gives any fixed-width neural payload a TrustMark-style data-bit profile: ``k``
data bits are packed with BCH parity into the model's ``raw_bits`` codeword, and
decoding silently corrects up to ``t`` bit-errors. Fewer data bits => more parity
=> more robust exact-ID. Picks the largest ``t`` (most correction) whose codeword
still fits ``raw_bits``.
"""

from __future__ import annotations


def bits_to_bytes(bits: list[int]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, v in enumerate(bits):
        if int(v) & 1:
            out[i // 8] |= 1 << (7 - i % 8)
    return bytes(out)


def bytes_to_bits(data: bytes, n: int) -> list[int]:
    return [(data[i // 8] >> (7 - i % 8)) & 1 for i in range(n)]


class BchCodec:
    """BCH(m=8) sized so ``data_bits`` + parity fit within ``raw_bits``."""

    def __init__(self, data_bits: int, raw_bits: int) -> None:
        import bchlib
        self.data_bits = int(data_bits)
        self.raw_bits = int(raw_bits)
        self.dbytes = (self.data_bits + 7) // 8
        self.bch = None
        self.t = None
        for t in range(1, 40):
            try:
                cand = bchlib.BCH(t, m=8)
            except Exception:  # pragma: no cover - invalid t
                continue
            if (self.dbytes + cand.ecc_bytes) * 8 <= self.raw_bits:
                self.bch, self.t = cand, t  # keep largest fitting t
        if self.bch is None:
            raise ValueError(
                f"no BCH(m=8) fits {data_bits} data bits in {raw_bits} raw bits")

    def encode(self, message: list[int]) -> list[int]:
        """``data_bits`` message -> ``raw_bits`` codeword (zero-padded)."""
        data = bits_to_bytes([int(b) & 1 for b in message[: self.data_bits]])
        data = (data + bytes(self.dbytes))[: self.dbytes]
        ecc = bytes(self.bch.encode(data))
        code = (bytes_to_bits(data, self.dbytes * 8)
                + bytes_to_bits(ecc, self.bch.ecc_bytes * 8))
        return code + [0] * (self.raw_bits - len(code))

    def decode(self, raw_bits: list[int]) -> tuple[list[int], bool]:
        """recovered ``raw_bits`` -> (``data_bits`` message, decoded_ok)."""
        db, eb = self.dbytes, self.bch.ecc_bytes
        data = bytearray(bits_to_bytes(raw_bits[: db * 8])[:db])
        ecc = bytearray(bits_to_bytes(raw_bits[db * 8: (db + eb) * 8])[:eb])
        try:
            nerr = self.bch.decode(data, ecc)
            self.bch.correct(data, ecc)
            ok = nerr is not None and nerr >= 0
        except Exception:  # pragma: no cover - uncorrectable
            ok = False
        return bytes_to_bits(bytes(data), self.data_bits), ok
