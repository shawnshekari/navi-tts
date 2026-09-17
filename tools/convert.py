#!/usr/bin/env python3
"""HF Qwen3-TTS safetensors -> .navi (docs/navi-format.md).

    uv run convert.py ../models/Qwen3-TTS-12Hz-0.6B-Base ../models/qwen3-tts-0.6b-f16.navi

Depends on numpy only: safetensors is a JSON header plus raw bytes, and bf16 is
an f32 with the low half dropped, so neither torch nor the safetensors package
is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
PAGE = 4096

DTYPE_F32, DTYPE_F16, DTYPE_BF16, DTYPE_I32, DTYPE_U8 = 0, 1, 2, 3, 4
KV_I64, KV_F64, KV_STR, KV_I64_ARRAY = 0, 1, 2, 3

NP_DTYPE = {DTYPE_F32: np.float32, DTYPE_F16: np.float16, DTYPE_I32: np.int32, DTYPE_U8: np.uint8}
TARGET_DTYPES = {"f16": DTYPE_F16, "f32": DTYPE_F32}


# --- safetensors ------------------------------------------------------------

class SafeTensors:
    """Read-only view over one safetensors file (memory-mapped)."""

    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        self.meta = header.pop("__metadata__", None)
        self.entries = header
        self.data_start = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return list(self.entries.keys())

    def get(self, name: str) -> np.ndarray:
        e = self.entries[name]
        a, b = e["data_offsets"]
        raw = self.mm[self.data_start + a : self.data_start + b]
        shape = tuple(e["shape"])
        dt = e["dtype"]
        if dt == "F32":
            return raw.view(np.float32).reshape(shape)
        if dt == "F16":
            return raw.view(np.float16).reshape(shape)
        if dt == "BF16":
            u = raw.view(np.uint16).astype(np.uint32) << 16
            return u.view(np.float32).reshape(shape)
        if dt == "I64":
            return raw.view(np.int64).reshape(shape)
        if dt == "I32":
            return raw.view(np.int32).reshape(shape)
        raise ValueError(f"{name}: unsupported safetensors dtype {dt}")


# --- .navi writer -----------------------------------------------------------

def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


class NaviWriter:
    def __init__(self):
        self.kv: list[tuple[str, int, object]] = []
        self.tensors: list[tuple[str, int, tuple[int, ...], np.ndarray]] = []
        self._names: set[str] = set()

    def add_kv(self, key: str, value):
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            self.kv.append((key, KV_I64, value))
        elif isinstance(value, float):
            self.kv.append((key, KV_F64, value))
        elif isinstance(value, str):
            self.kv.append((key, KV_STR, value))
        elif isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value):
            self.kv.append((key, KV_I64_ARRAY, list(value)))
        elif value is None:
            pass
        else:
            raise TypeError(f"{key}: cannot store {type(value).__name__} in a .navi KV")

    def add_tensor(self, name: str, dtype: int, arr: np.ndarray):
        if name in self._names:
            raise ValueError(f"duplicate tensor {name}")
        self._names.add(name)
        arr = np.ascontiguousarray(arr.astype(NP_DTYPE[dtype], copy=False))
        self.tensors.append((name, dtype, tuple(int(d) for d in arr.shape), arr))

    def write(self, path: Path):
        # tensor table with offsets
        offsets, off = [], 0
        for _, _, _, arr in self.tensors:
            offsets.append(off)
            off += (arr.nbytes + PAGE - 1) // PAGE * PAGE
        data_size = off

        table = bytearray()
        for key, t, v in self.kv:
            table += _string(key) + struct.pack("<B", t)
            if t == KV_I64:
                table += struct.pack("<q", v)
            elif t == KV_F64:
                table += struct.pack("<d", v)
            elif t == KV_STR:
                table += _string(v)
            else:
                table += struct.pack("<I", len(v)) + b"".join(struct.pack("<q", x) for x in v)
        for (name, dtype, shape, arr), o in zip(self.tensors, offsets):
            table += _string(name) + struct.pack("<BB", dtype, len(shape))
            table += b"".join(struct.pack("<Q", d) for d in shape)
            table += struct.pack("<QQ", o, arr.nbytes)

        header_len = 32 + len(table)
        data_offset = (header_len + PAGE - 1) // PAGE * PAGE
        file_size = data_offset + data_size
        head = b"NAVI" + struct.pack("<IQIIQ", FORMAT_VERSION, data_offset, len(self.kv), len(self.tensors), file_size)

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(head)
            f.write(table)
            f.write(b"\0" * (data_offset - header_len))
            for (name, dtype, shape, arr), o in zip(self.tensors, offsets):
                assert f.tell() == data_offset + o, name
                f.write(arr.tobytes())
                pad = (arr.nbytes + PAGE - 1) // PAGE * PAGE - arr.nbytes
                if pad:
                    f.write(b"\0" * pad)
            assert f.tell() == file_size
        os.replace(tmp, path)
        return file_size


# --- Qwen3-TTS specifics ----------------------------------------------------

def flatten_config(w: NaviWriter, prefix: str, cfg: dict, keys: list[str]):
    for k in keys:
        if k in cfg and cfg[k] is not None:
            w.add_kv(f"{prefix}.{k}", cfg[k])


TRANSFORMER_KEYS = [
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
    "intermediate_size", "rms_norm_eps", "rope_theta", "vocab_size", "max_position_embeddings",
    "hidden_act", "attention_bias",
]


def write_metadata(w: NaviWriter, root: Path, with_encoder: bool, target: str):
    cfg = json.loads((root / "config.json").read_text())
    tk = cfg["talker_config"]
    cp = tk["code_predictor_config"]
    codec = json.loads((root / "speech_tokenizer" / "config.json").read_text())
    dec = codec["decoder_config"]
    gen = json.loads((root / "generation_config.json").read_text())

    w.add_kv("general.arch", "qwen3-tts")
    w.add_kv("general.name", root.name)
    w.add_kv("general.model_type", cfg["model_type"])
    w.add_kv("general.tts_model_type", cfg["tts_model_type"])
    w.add_kv("general.tts_model_size", cfg["tts_model_size"])
    w.add_kv("general.tokenizer_type", cfg["tokenizer_type"])
    w.add_kv("general.dtype", target)
    w.add_kv("general.converter", "navi-tts tools/convert.py format-v1")
    w.add_kv("general.has_codec_encoder", with_encoder)

    for k in ("im_start_token_id", "im_end_token_id", "assistant_token_id",
              "tts_bos_token_id", "tts_eos_token_id", "tts_pad_token_id"):
        w.add_kv("text." + k.replace("_token_id", "_id"), cfg[k])

    flatten_config(w, "talker", tk, TRANSFORMER_KEYS + [
        "text_hidden_size", "text_vocab_size", "num_code_groups", "position_id_per_seconds",
        "codec_bos_id", "codec_eos_token_id", "codec_pad_id", "codec_think_id", "codec_nothink_id",
        "codec_think_bos_id", "codec_think_eos_id",
    ])
    rs = tk.get("rope_scaling") or {}
    w.add_kv("talker.mrope_section", rs.get("mrope_section"))
    w.add_kv("talker.mrope_interleaved", bool(rs.get("interleaved", False)))
    for name, lid in tk["codec_language_id"].items():
        w.add_kv(f"talker.language.{name}", lid)
    for name, sid in tk.get("spk_id", {}).items():
        w.add_kv(f"talker.speaker.{name}", sid)

    flatten_config(w, "cp", cp, TRANSFORMER_KEYS + ["num_code_groups"])

    # config.json carries only enc_dim and sample_rate; the rest are the model
    # class's defaults (Qwen3TTSSpeakerEncoderConfig), written so the blob is self-describing.
    se = dict(mel_dim=128, enc_channels=[512, 512, 512, 512, 1536], enc_kernel_sizes=[5, 3, 3, 3, 1],
              enc_dilations=[1, 2, 3, 4, 1], enc_attention_channels=128, enc_res2net_scale=8,
              enc_se_channels=128, enc_dim=192, sample_rate=24000)
    se.update(cfg["speaker_encoder_config"])
    for k, v in se.items():
        w.add_kv(f"speaker.{k}", v)
    # The mel front end is fixed in the model's extract_speaker_embedding (not in config.json).
    for k, v in dict(n_fft=1024, num_mels=128, hop_size=256, win_size=1024, fmin=0, fmax=12000).items():
        w.add_kv(f"speaker.mel.{k}", v)

    w.add_kv("codec.input_sample_rate", codec["input_sample_rate"])
    w.add_kv("codec.output_sample_rate", codec["output_sample_rate"])
    w.add_kv("codec.decode_upsample_rate", codec["decode_upsample_rate"])
    w.add_kv("codec.encode_downsample_rate", codec["encode_downsample_rate"])
    w.add_kv("codec.encoder_valid_num_quantizers", codec["encoder_valid_num_quantizers"])
    flatten_config(w, "codec.decoder", dec, TRANSFORMER_KEYS + [
        "latent_dim", "codebook_dim", "codebook_size", "decoder_dim", "num_quantizers",
        "num_semantic_quantizers", "semantic_codebook_size", "sliding_window",
        "vector_quantization_hidden_dimension", "layer_scale_initial_scale",
        "upsample_rates", "upsampling_ratios",
    ])
    if with_encoder:
        enc = codec["encoder_config"]
        flatten_config(w, "codec.encoder", enc, [
            "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "intermediate_size", "norm_eps", "rope_theta", "max_position_embeddings", "hidden_act",
            "codebook_dim", "codebook_size", "num_quantizers", "num_semantic_quantizers", "num_filters",
            "kernel_size", "last_kernel_size", "residual_kernel_size", "compress", "dilation_growth_rate",
            "num_residual_layers", "sliding_window", "upsampling_ratios", "upsample_groups",
            "vector_quantization_hidden_dimension", "layer_scale_initial_scale", "use_causal_conv",
            "pad_mode", "trim_right_ratio", "sampling_rate",
        ])

    for k in ("temperature", "top_k", "top_p", "repetition_penalty", "do_sample",
              "subtalker_temperature", "subtalker_top_k", "subtalker_top_p", "subtalker_dosample"):
        if k in gen:
            w.add_kv(f"gen.{k}", gen[k])


def add_file_bytes(w: NaviWriter, name: str, path: Path):
    w.add_tensor(name, DTYPE_U8, np.frombuffer(path.read_bytes(), dtype=np.uint8))


class Stats:
    def __init__(self):
        self.n = 0
        self.bytes = 0
        self.overflow = 0
        self.overflow_names: list[str] = []


def store(w: NaviWriter, st: Stats, name: str, arr: np.ndarray, target_dtype: int):
    # >=2-D in the target dtype; 1-D (norms, biases, scales) stays f32
    dtype = target_dtype if arr.ndim >= 2 else DTYPE_F32
    if dtype == DTYPE_F16:
        a = np.abs(arr[np.isfinite(arr)])
        n_over = int((a > 65504.0).sum()) if a.size else 0
        if n_over:
            st.overflow += n_over
            st.overflow_names.append(name)
    w.add_tensor(name, dtype, arr)
    st.n += 1
    st.bytes += arr.size * np.dtype(NP_DTYPE[dtype]).itemsize


def convert(root: Path, out: Path, target: str, with_encoder: bool, verbose: bool) -> None:
    target_dtype = TARGET_DTYPES[target]
    w = NaviWriter()
    write_metadata(w, root, with_encoder, target)
    for fname, tname in (("vocab.json", "tokenizer.vocab.json"), ("merges.txt", "tokenizer.merges.txt"),
                         ("tokenizer_config.json", "tokenizer.config.json")):
        add_file_bytes(w, tname, root / fname)

    st = Stats()
    main = SafeTensors(root / "model.safetensors")
    w.add_kv("general.source.main", "model.safetensors")
    for name in main.names():
        store(w, st, name, main.get(name), target_dtype)
        if verbose:
            print(f"  {name}", file=sys.stderr)

    codec = SafeTensors(root / "speech_tokenizer" / "model.safetensors")
    w.add_kv("general.source.codec", "speech_tokenizer/model.safetensors")
    names = codec.names()
    pending_codebooks: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        if name.startswith("encoder.") and not with_encoder:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "initialized":
            continue
        if leaf in ("embedding_sum", "embed_sum", "cluster_usage"):
            base = name.rsplit(".", 1)[0]
            pending_codebooks.setdefault(base, {})[leaf] = codec.get(name)
            continue
        store(w, st, "codec." + name, codec.get(name), target_dtype)
    for base, parts in pending_codebooks.items():
        emb = parts.get("embedding_sum", parts.get("embed_sum"))
        usage = parts["cluster_usage"]
        # EuclideanCodebook.decode: embedding_sum / cluster_usage.clamp(min=1e-5)
        book = emb / np.maximum(usage, 1e-5)[:, None]
        store(w, st, "codec." + base + ".codebook", book.astype(np.float32), target_dtype)

    size = w.write(out)
    print(f"wrote {out} ({size / 1e9:.3f} GB): {len(w.kv)} kv, {len(w.tensors)} tensors, "
          f"{st.bytes / 1e9:.3f} GB of weights as {target}", file=sys.stderr)
    if st.overflow:
        print(f"WARNING: {st.overflow} values exceeded f16 range in {len(st.overflow_names)} tensors: "
              f"{st.overflow_names[:8]}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path, help="HF model directory (config.json, model.safetensors, speech_tokenizer/)")
    ap.add_argument("out", type=Path, help="output .navi path")
    ap.add_argument("--dtype", choices=sorted(TARGET_DTYPES), default="f16", help="dtype for >=2-D tensors (default f16)")
    ap.add_argument("--with-encoder", action="store_true", help="include the codec encoder (ICL cloning)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    convert(a.model_dir, a.out, a.dtype, a.with_encoder, a.verbose)


if __name__ == "__main__":
    main()
