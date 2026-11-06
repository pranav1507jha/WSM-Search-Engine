# compression.py
import math
from typing import List, Dict, Tuple, Optional


class BlockedDictionary:
    """
    Compresses the vocabulary using Dictionary-as-a-String with Blocking (k=4).
    Each term in the concatenated string is prefixed by a 1-byte length character.
    Only 1 pointer is stored per block of k=4 terms.
    """
    def __init__(self, sorted_terms: List[str], block_size: int = 4):
        self.block_size = block_size
        self.num_terms = len(sorted_terms)
        self.concat_string = ""
        self.block_pointers: List[int] = []
        self._build(sorted_terms)

    def _build(self, sorted_terms: List[str]):
        chunks = []
        current_offset = 0
        for idx, term in enumerate(sorted_terms):
            if idx % self.block_size == 0:
                self.block_pointers.append(current_offset)
            encoded_len = chr(min(len(term), 255))
            entry = encoded_len + term
            chunks.append(entry)
            current_offset += len(entry)
        self.concat_string = "".join(chunks)

    def lookup(self, target: str) -> Optional[int]:
        """Binary search over block pointers, followed by linear scan of up to k terms."""
        low = 0
        high = len(self.block_pointers) - 1

        while low <= high:
            mid = (low + high) // 2
            ptr = self.block_pointers[mid]
            first_len = ord(self.concat_string[ptr])
            first_term = self.concat_string[ptr + 1: ptr + 1 + first_len]

            if target == first_term:
                return mid * self.block_size
            elif target < first_term:
                high = mid - 1
            else:
                # Check if target could be inside this block or a later block
                if mid + 1 < len(self.block_pointers):
                    next_ptr = self.block_pointers[mid + 1]
                    next_len = ord(self.concat_string[next_ptr])
                    next_term = self.concat_string[next_ptr + 1: next_ptr + 1 + next_len]
                    if target < next_term:
                        return self._scan_block(mid, target)
                    else:
                        low = mid + 1
                else:
                    return self._scan_block(mid, target)
        return None

    def _scan_block(self, block_idx: int, target: str) -> Optional[int]:
        ptr = self.block_pointers[block_idx]
        end_ptr = (
            self.block_pointers[block_idx + 1]
            if block_idx + 1 < len(self.block_pointers)
            else len(self.concat_string)
        )
        offset = 0
        while ptr < end_ptr and offset < self.block_size:
            t_len = ord(self.concat_string[ptr])
            term = self.concat_string[ptr + 1: ptr + 1 + t_len]
            if term == target:
                return block_idx * self.block_size + offset
            ptr += 1 + t_len
            offset += 1
        return None

    def byte_size(self) -> int:
        # String bytes + 3 bytes per block pointer + 8 bytes metadata (df + postings ptr) per term
        string_bytes = len(self.concat_string.encode("utf-8"))
        pointer_bytes = len(self.block_pointers) * 3
        metadata_bytes = self.num_terms * 8
        return string_bytes + pointer_bytes + metadata_bytes


class FrontCodedDictionary:
    """
    Compresses blocks of k=4 sorted terms by factoring out the common prefix
    within each block (e.g., 8{automat}a1<>e2<>ic3<>ion).
    """
    def __init__(self, sorted_terms: List[str], block_size: int = 4):
        self.block_size = block_size
        self.num_terms = len(sorted_terms)
        self.encoded_blocks: List[str] = []
        self._build(sorted_terms)

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        limit = min(len(a), len(b))
        i = 0
        while i < limit and a[i] == b[i]:
            i += 1
        return i

    def _build(self, sorted_terms: List[str]):
        for i in range(0, len(sorted_terms), self.block_size):
            block = sorted_terms[i: i + self.block_size]
            first = block[0]
            rep = f"{len(first)}{first}"
            for term in block[1:]:
                p_len = self._common_prefix_len(first, term)
                suffix = term[p_len:]
                rep += f"*{p_len}:{suffix}"
            self.encoded_blocks.append(rep)

    def byte_size(self) -> int:
        raw_str = "".join(self.encoded_blocks)
        string_bytes = len(raw_str.encode("utf-8"))
        pointer_bytes = len(self.encoded_blocks) * 3
        metadata_bytes = self.num_terms * 8
        return string_bytes + pointer_bytes + metadata_bytes


class VariableByteCodec:
    """
    Encodes sorted document ID lists as gaps using byte-aligned Variable-Byte (VB) codes.
    Low 7 bits store data; high bit (0x80) is the continuation bit (1 on last byte of gap).
    """
    @staticmethod
    def to_gaps(doc_ids: List[int]) -> List[int]:
        if not doc_ids:
            return []
        gaps = [doc_ids[0]]
        for i in range(1, len(doc_ids)):
            gaps.append(doc_ids[i] - doc_ids[i - 1])
        return gaps

    @staticmethod
    def from_gaps(gaps: List[int]) -> List[int]:
        if not gaps:
            return []
        doc_ids = [gaps[0]]
        for i in range(1, len(gaps)):
            doc_ids.append(doc_ids[-1] + gaps[i])
        return doc_ids

    @staticmethod
    def encode_number(n: int) -> bytearray:
        bytes_list = bytearray()
        while True:
            bytes_list.insert(0, n % 128)
            if n < 128:
                break
            n //= 128
        bytes_list[-1] += 128  # Set continuation termination bit on final byte
        return bytes_list

    @classmethod
    def encode(cls, doc_ids: List[int]) -> bytes:
        gaps = cls.to_gaps(doc_ids)
        stream = bytearray()
        for g in gaps:
            stream.extend(cls.encode_number(g))
        return bytes(stream)

    @classmethod
    def decode(cls, bytestream: bytes) -> List[int]:
        gaps = []
        n = 0
        for b in bytestream:
            if b < 128:
                n = 128 * n + b
            else:
                n = 128 * n + (b - 128)
                gaps.append(n)
                n = 0
        return cls.from_gaps(gaps)


class EliasGammaCodec:
    """
    Encodes positive integer gaps G as <length, offset>:
    - length: floor(log2(G)) ones followed by a 0 bit (unary)
    - offset: G - 2^floor(log2(G)) in floor(log2(G)) binary bits.
    """
    @staticmethod
    def encode_gap(g: int) -> str:
        if g <= 0:
            g = 1
        k = int(math.floor(math.log2(g)))
        unary_length = ("1" * k) + "0"
        offset_val = g - (1 << k)
        binary_offset = format(offset_val, f"0{k}b") if k > 0 else ""
        return unary_length + binary_offset

    @classmethod
    def encode(cls, doc_ids: List[int]) -> Tuple[str, int]:
        gaps = VariableByteCodec.to_gaps(doc_ids)
        bitstring = "".join(cls.encode_gap(g) for g in gaps)
        byte_len = math.ceil(len(bitstring) / 8.0)
        return bitstring, byte_len

    @classmethod
    def decode(cls, bitstring: str) -> List[int]:
        gaps = []
        i = 0
        n = len(bitstring)
        while i < n:
            k = 0
            while i < n and bitstring[i] == "1":
                k += 1
                i += 1
            i += 1  # Skip separating '0'
            if k == 0:
                gaps.append(1)
            else:
                offset_bits = bitstring[i: i + k]
                i += k
                offset_val = int(offset_bits, 2) if offset_bits else 0
                gaps.append((1 << k) + offset_val)
        return VariableByteCodec.from_gaps(gaps)


def compute_compression_report(sorted_terms: List[str], postings_map: Dict[str, List[int]]) -> dict:
    """Computes live storage metrics comparing uncompressed vs compressed structures."""
    num_terms = len(sorted_terms)
    # Fixed-width dictionary: 20 bytes term + 4 bytes df + 4 bytes postings ptr = 28 bytes/term
    fixed_dict_bytes = num_terms * 28

    # Uncompressed dictionary string (3B term ptr + 4B df + 4B postings ptr + avg char len)
    unblocked_str_bytes = sum(len(t.encode("utf-8")) for t in sorted_terms) + num_terms * 11

    blocked_dict = BlockedDictionary(sorted_terms, block_size=4)
    blocked_bytes = blocked_dict.byte_size()

    front_coded_dict = FrontCodedDictionary(sorted_terms, block_size=4)
    front_coded_bytes = front_coded_dict.byte_size()

    # Postings comparison
    total_postings = sum(len(pids) for pids in postings_map.values())
    uncompressed_postings_bytes = total_postings * 4  # 4-byte (32-bit) integers

    vb_bytes = 0
    gamma_bits = 0
    sample_encodings = []

    for idx, term in enumerate(sorted_terms):
        doc_ids = sorted(postings_map.get(term, []))
        vb_stream = VariableByteCodec.encode(doc_ids)
        vb_bytes += len(vb_stream)
        g_bitstr, _ = EliasGammaCodec.encode(doc_ids)
        gamma_bits += len(g_bitstr)

        if len(doc_ids) >= 3 and len(sample_encodings) < 5:
            gaps = VariableByteCodec.to_gaps(doc_ids)
            sample_encodings.append({
                "term": term,
                "doc_ids": doc_ids,
                "gaps": gaps,
                "vb_hex": vb_stream.hex(),
                "elias_gamma_bits": g_bitstr
            })

    gamma_bytes = math.ceil(gamma_bits / 8.0)

    return {
        "dictionary": {
            "num_terms": num_terms,
            "fixed_width_bytes": fixed_dict_bytes,
            "dictionary_as_string_bytes": unblocked_str_bytes,
            "blocked_k4_bytes": blocked_bytes,
            "front_coded_k4_bytes": front_coded_bytes,
            "blocked_savings_pct": round((1 - blocked_bytes / max(fixed_dict_bytes, 1)) * 100, 1),
            "front_coded_savings_pct": round((1 - front_coded_bytes / max(fixed_dict_bytes, 1)) * 100, 1)
        },
        "postings": {
            "total_postings": total_postings,
            "uncompressed_bytes": uncompressed_postings_bytes,
            "variable_byte_bytes": vb_bytes,
            "elias_gamma_bytes": gamma_bytes,
            "elias_gamma_bits": gamma_bits,
            "vb_savings_pct": round((1 - vb_bytes / max(uncompressed_postings_bytes, 1)) * 100, 1),
            "gamma_savings_pct": round((1 - gamma_bytes / max(uncompressed_postings_bytes, 1)) * 100, 1)
        },
        "samples": sample_encodings
    }