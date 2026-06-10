#!/usr/bin/env python3
"""Bidirectional token ID matcher between Ouro-1.4B and HRM-Text-1B.

Maps token IDs across differing BPE vocabularies with confidence levels.
Handles special tokens via explicit mapping, falls back to decode/re-encode
for regular tokens, and validates round-trips.

Usage:
  token_matcher.py                # interactive REPL
  echo "ouro 42" | token_matcher.py
  echo "hrm 100 200 300" | token_matcher.py
  echo "seq ouro 12 45 88 2" | token_matcher.py
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tokenizers import Tokenizer


BASE = Path(__file__).parent.resolve()
MODEL_DIRS = {"ouro": "Ouro-1.4B", "hrm": "HRM-Text-1B"}


@dataclass
class Match:
    confidence: str  # "exact" | "approx" | "mismatch" | "invalid"
    target_ids: list[int]
    source_str: str | None = None
    target_str: str | None = None
    note: str = ""


def _load_added_tokens(path: Path) -> dict[int, str]:
    with open(path) as f:
        data = json.load(f)
    return {t["id"]: t["content"] for t in data.get("added_tokens", [])}


class TokenMatcher:
    def __init__(self):
        self._check_dirs()
        self.ouro_tok = Tokenizer.from_file(str(BASE / "Ouro-1.4B/tokenizer.json"))
        self.hrm_tok = Tokenizer.from_file(str(BASE / "HRM-Text-1B/tokenizer.json"))

        self.hrm_vocab = self.hrm_tok.get_vocab()
        self.ouro_vocab = self.ouro_tok.get_vocab()

        self.hrm_id_to_str = {v: k for k, v in self.hrm_vocab.items()}
        self.ouro_id_to_str = {v: k for k, v in self.ouro_vocab.items()}

        self.ouro_special = _load_added_tokens(BASE / "Ouro-1.4B/tokenizer.json")
        self.hrm_special = _load_added_tokens(BASE / "HRM-Text-1B/tokenizer.json")

    @staticmethod
    def _check_dirs():
        missing = []
        for name in MODEL_DIRS.values():
            if not (BASE / name / "tokenizer.json").exists():
                missing.append(name)
        if missing:
            paths = ", ".join(f"{BASE}/{m}/tokenizer.json" for m in missing)
            raise FileNotFoundError(
                f"Missing tokenizer dirs: {paths}. "
                f"Run this script from the directory containing Ouro-1.4B/ and HRM-Text-1B/."
            )

    def _is_special(self, token_id: int, src: str) -> bool:
        specials = self.ouro_special if src == "ouro" else self.hrm_special
        return token_id in specials

    def _special_crosswalk(self, token_str: str, src: str, dst: str) -> int | None:
        src_specials = self.ouro_special if src == "ouro" else self.hrm_special
        src_by_str = {v: k for k, v in src_specials.items()}
        dst_specials = self.hrm_special if dst == "hrm" else self.ouro_special
        dst_by_str = {v: k for k, v in dst_specials.items()}

        if token_str in src_by_str:
            return dst_by_str.get(token_str)
        return None

    def _round_trip_check(self, src: str, src_id: int, target_ids: list[int]) -> tuple[bool, str]:
        source_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        target_tok = self.hrm_tok if src == "ouro" else self.ouro_tok
        dst_name = "hrm" if src == "ouro" else "ouro"

        original_str = source_tok.decode([src_id], skip_special_tokens=False)
        target_str = target_tok.decode(target_ids, skip_special_tokens=False)

        if original_str == target_str:
            return True, ""

        re_encoded = source_tok.encode(target_str).ids
        if src_id in re_encoded:
            return True, (
                f"string mismatch ({original_str!r} vs {target_str!r}) "
                f"but src_id recovered in re-encode"
            )

        return False, (
            f"decode({original_str!r}) -> {dst_name}{target_ids} "
            f"-> decode({target_str!r}) -> re-encode -> {re_encoded} "
            f"(expected to contain {src_id})"
        )

    def _normalize_candidates(self, s: str, src: str) -> list[str]:
        candidates = [s]
        if src == "ouro" and s.startswith("\u0120"):
            candidates.append(s[1:])
            candidates.append(" " + s[1:])
        return candidates

    def _map_single(self, token_id: int, src: str, dst: str) -> Match:
        src_vocab = self.ouro_vocab if src == "ouro" else self.hrm_vocab
        dst_vocab = self.hrm_vocab if src == "ouro" else self.ouro_vocab
        src_id_to_str = self.ouro_id_to_str if src == "ouro" else self.hrm_id_to_str
        src_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        dst_tok = self.hrm_tok if src == "ouro" else self.ouro_tok
        dst_name = "hrm" if src == "ouro" else "ouro"

        token_str = src_id_to_str.get(token_id)
        if token_str is None:
            return Match("invalid", [], note=f"token ID {token_id} not in {src} vocab")

        if self._is_special(token_id, src):
            target_id = self._special_crosswalk(token_str, src, dst)
            if target_id is not None:
                return Match(
                    "exact", [target_id],
                    source_str=token_str,
                    target_str=dst_tok.decode([target_id], skip_special_tokens=False),
                    note="special token",
                )

        for candidate in self._normalize_candidates(token_str, src):
            if candidate in dst_vocab:
                tid = dst_vocab[candidate]
                return Match("exact", [tid], source_str=token_str, target_str=candidate)

        decoded = src_tok.decode([token_id], skip_special_tokens=False)
        target_ids = dst_tok.encode(decoded).ids

        if not target_ids:
            return Match(
                "approx", [],
                source_str=token_str,
                target_str="",
                note=f"encode({decoded!r}) -> empty",
            )

        ok, msg = self._round_trip_check(src, token_id, target_ids)
        confidence = "approx" if ok else "mismatch"
        target_str = dst_tok.decode(target_ids, skip_special_tokens=False)
        return Match(confidence, target_ids, source_str=token_str, target_str=target_str, note=msg)

    def ouro_to_hrm(self, token_id: int) -> Match:
        return self._map_single(token_id, "ouro", "hrm")

    def hrm_to_ouro(self, token_id: int) -> Match:
        return self._map_single(token_id, "hrm", "ouro")

    def map_sequence(self, token_ids: list[int], src: str) -> Match:
        src_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        dst_tok = self.hrm_tok if src == "ouro" else self.ouro_tok
        dst_name = "hrm" if src == "ouro" else "ouro"

        decoded = src_tok.decode(token_ids, skip_special_tokens=False)
        target_ids = dst_tok.encode(decoded).ids

        if not target_ids:
            return Match("approx", [], source_str=decoded, target_str="",
                         note=f"encode({decoded!r}) -> empty")

        target_str = dst_tok.decode(target_ids, skip_special_tokens=False)
        re_encoded = src_tok.encode(target_str).ids

        if decoded == target_str:
            confidence = "exact"
        elif any(tid in re_encoded for tid in token_ids):
            confidence = "approx"
        else:
            confidence = "mismatch"

        return Match(confidence, target_ids, source_str=decoded, target_str=target_str)

    def format_match(self, m: Match, src_name: str, src_id: int | None = None) -> str:
        label = src_name.upper()
        if src_id is not None:
            label += f" [{src_id}]"
        if m.source_str:
            label += f" ({m.source_str!r})"

        if m.confidence == "invalid":
            return f"  {label} -> <{m.note}>"

        parts = " + ".join(f"[{tid}]" for tid in m.target_ids)
        conf_mark = {"exact": "✓", "approx": "~", "mismatch": "✗"}.get(m.confidence, "?")
        result = f"  {label} -> {parts} [{conf_mark}]"

        if m.note:
            result += f"  # {m.note}"

        return result

    def show_info(self):
        print("Token Matcher: Ouro-1.4B <-> HRM-Text-1B")
        print(f"  Ouro vocab: {len(self.ouro_vocab):>5}  ({len(self.ouro_special):>2} special)")
        print(f"  HRM  vocab: {len(self.hrm_vocab):>5}  ({len(self.hrm_special):>2} special)")
        print(f"  Shared special tokens: "
              f"{len(set(self.ouro_special.values()) & set(self.hrm_special.values()))}")
        print()


def interactive(matcher: TokenMatcher):
    matcher.show_info()
    while True:
        try:
            inp = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not inp:
            continue
        if inp.lower() in ("q", "quit", "exit"):
            break

        parts = inp.split()
        if len(parts) < 2:
            print("  Usage: <model> <id> [...] | seq <model> <id> [...]")
            continue

        if parts[0].lower() == "seq":
            model = parts[1].lower()
            ids = parts[2:]
            if model not in ("ouro", "hrm"):
                print(f"  Unknown model '{model}'. Use 'ouro' or 'hrm'.")
                continue
            try:
                token_ids = [int(x) for x in ids]
            except ValueError:
                print(f"  Invalid token IDs: {ids}")
                continue
            m = matcher.map_sequence(token_ids, model)
            print(matcher.format_match(m, model.upper()))
        else:
            model = parts[0].lower()
            ids = parts[1:]
            if model not in ("ouro", "hrm"):
                print(f"  Unknown model '{model}'. Use 'ouro' or 'hrm'.")
                continue
            for sid in ids:
                try:
                    tid = int(sid)
                except ValueError:
                    print(f"  Invalid token ID: {sid}")
                    continue
                fn = matcher.ouro_to_hrm if model == "ouro" else matcher.hrm_to_ouro
                m = fn(tid)
                print(matcher.format_match(m, model, tid))


def main():
    try:
        matcher = TokenMatcher()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) > 1:
        matcher.show_info()
        raw = sys.argv[1]
        parts = raw.split()
        if parts[0].lower() == "seq":
            model = parts[1].lower()
            ids = [int(x) for x in parts[2:]]
            m = matcher.map_sequence(ids, model)
            print(matcher.format_match(m, model.upper()))
        else:
            model = parts[0].lower()
            ids = [int(x) for x in parts[1:]]
            for tid in ids:
                fn = matcher.ouro_to_hrm if model == "ouro" else matcher.hrm_to_ouro
                m = fn(tid)
                print(matcher.format_match(m, model, tid))
    else:
        interactive(matcher)


if __name__ == "__main__":
    main()
