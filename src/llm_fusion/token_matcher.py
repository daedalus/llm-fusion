"""Bidirectional token ID matcher between Ouro-1.4B and HRM-Text-1B."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tokenizers import Tokenizer

log = logging.getLogger(__name__)


@dataclass
class Match:
    confidence: Literal["exact", "approx", "mismatch", "invalid"]
    target_ids: list[int]
    source_str: str | None = None
    target_str: str | None = None
    note: str = ""


def _load_added_tokens(path: Path) -> dict[int, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[int, str] = {}
    for t in data.get("added_tokens", []):
        try:
            result[t["id"]] = t["content"]
        except KeyError as e:
            raise ValueError(
                f"Missing key {e} in added token entry {t} from {path}"
            ) from e
    return result


class TokenMatcher:
    def __init__(
        self,
        ouro_tokenizer_path: str | Path = "Ouro-1.4B/tokenizer.json",
        hrm_tokenizer_path: str | Path = "HRM-Text-1B/tokenizer.json",
    ) -> None:
        self.ouro_path = Path(ouro_tokenizer_path)
        self.hrm_path = Path(hrm_tokenizer_path)
        self._check_dirs()
        self.ouro_tok = Tokenizer.from_file(str(self.ouro_path))
        self.hrm_tok = Tokenizer.from_file(str(self.hrm_path))

        self.hrm_vocab = self.hrm_tok.get_vocab()
        self.ouro_vocab = self.ouro_tok.get_vocab()

        self.hrm_id_to_str = {v: k for k, v in self.hrm_vocab.items()}
        self.ouro_id_to_str = {v: k for k, v in self.ouro_vocab.items()}

        self.ouro_special = _load_added_tokens(self.ouro_path)
        self.hrm_special = _load_added_tokens(self.hrm_path)

    def _check_dirs(self) -> None:
        missing = []
        for p in [self.ouro_path, self.hrm_path]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(f"Missing tokenizer files: {', '.join(missing)}")

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

    @staticmethod
    def _normalize_bpe(s: str) -> str:
        return s.replace("\u0120", " ")

    def _round_trip_check(self, src: str, src_id: int, target_ids: list[int]) -> tuple[bool, str]:
        source_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        target_tok = self.hrm_tok if src == "ouro" else self.ouro_tok
        dst_name = "hrm" if src == "ouro" else "ouro"

        original_str = source_tok.decode([src_id], skip_special_tokens=False)
        target_str = target_tok.decode(target_ids, skip_special_tokens=False)

        if self._normalize_bpe(original_str) == self._normalize_bpe(target_str):
            return True, ""

        return False, (
            f"decode({original_str!r}) -> {dst_name} {target_ids} "
            f"-> decode({target_str!r})"
        )

    def _normalize_candidates(self, s: str) -> list[str]:
        candidates = [s]
        if s.startswith("\u0120"):
            candidates.append(s[1:])
            candidates.append(" " + s[1:])
        return candidates

    def _map_single(self, token_id: int, src: str, dst: str) -> Match:
        src_id_to_str = self.ouro_id_to_str if src == "ouro" else self.hrm_id_to_str
        dst_vocab = self.hrm_vocab if src == "ouro" else self.ouro_vocab
        src_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        dst_tok = self.hrm_tok if src == "ouro" else self.ouro_tok

        token_str = src_id_to_str.get(token_id)
        if token_str is None:
            return Match("invalid", [], note=f"token ID {token_id} not in {src} vocab")

        if self._is_special(token_id, src):
            target_id = self._special_crosswalk(token_str, src, dst)
            if target_id is not None:
                return Match(
                    "exact",
                    [target_id],
                    source_str=token_str,
                    target_str=dst_tok.decode([target_id], skip_special_tokens=False),
                    note="special token",
                )
            return Match(
                "mismatch",
                [],
                source_str=token_str,
                note=f"special token {token_str!r} has no {dst} equivalent",
            )

        for candidate in self._normalize_candidates(token_str):
            if candidate in dst_vocab:
                tid = dst_vocab[candidate]
                return Match("exact", [tid], source_str=token_str, target_str=candidate)

        decoded = src_tok.decode([token_id], skip_special_tokens=False)
        target_ids = dst_tok.encode(decoded).ids

        if not target_ids:
            log.warning(
                "_map_single %s->%s: encode(%r) returned empty for token %d",
                src, dst, decoded, token_id,
            )
            return Match(
                "approx",
                [],
                source_str=token_str,
                target_str="",
                note=f"encode({decoded!r}) -> empty",
            )

        ok, msg = self._round_trip_check(src, token_id, target_ids)
        target_str = dst_tok.decode(target_ids, skip_special_tokens=False)
        if ok:
            return Match("approx", target_ids, source_str=token_str, target_str=target_str, note=msg)
        return Match("mismatch", target_ids, source_str=token_str, target_str=target_str, note=msg)

    def ouro_to_hrm(self, token_id: int) -> Match:
        return self._map_single(token_id, "ouro", "hrm")

    def hrm_to_ouro(self, token_id: int) -> Match:
        return self._map_single(token_id, "hrm", "ouro")

    def _decode_encode_segment(
        self, token_ids: list[int], src_tok: Tokenizer, dst_tok: Tokenizer,
    ) -> tuple[list[int], str, str]:
        decoded = src_tok.decode(token_ids, skip_special_tokens=False)
        target_ids = dst_tok.encode(decoded).ids
        if not target_ids:
            return [], decoded, ""
        target_str = dst_tok.decode(target_ids, skip_special_tokens=False)
        return target_ids, decoded, target_str

    def map_sequence(self, token_ids: list[int], src: str) -> Match:
        src_tok = self.ouro_tok if src == "ouro" else self.hrm_tok
        dst_tok = self.hrm_tok if src == "ouro" else self.ouro_tok
        dst = "hrm" if src == "ouro" else "ouro"

        has_special = any(self._is_special(tid, src) for tid in token_ids)
        if not has_special:
            target_ids, decoded, target_str = self._decode_encode_segment(
                token_ids, src_tok, dst_tok,
            )
            if not target_ids:
                return Match(
                    "approx", [],
                    source_str=decoded, target_str="",
                    note=f"encode({decoded!r}) -> empty",
                )
            if len(target_ids) != len(token_ids):
                log.debug(
                    "map_sequence %s: token count changed %d -> %d for %r",
                    src, len(token_ids), len(target_ids), decoded,
                )
            if self._normalize_bpe(decoded) == self._normalize_bpe(target_str):
                return Match("exact", target_ids, source_str=decoded, target_str=target_str)
            return Match("mismatch", target_ids, source_str=decoded, target_str=target_str)

        rank = {"exact": 0, "approx": 1, "mismatch": 2, "invalid": 3}
        all_target_ids: list[int] = []
        run: list[int] = []
        overall: Literal["exact", "approx", "mismatch", "invalid"] = "exact"
        source_parts: list[str] = []
        target_parts: list[str] = []

        def _worse(
            a: Literal["exact", "approx", "mismatch", "invalid"],
            b: Literal["exact", "approx", "mismatch", "invalid"],
        ) -> Literal["exact", "approx", "mismatch", "invalid"]:
            return a if rank.get(a, 3) >= rank.get(b, 3) else b

        for tid in token_ids:
            if self._is_special(tid, src):
                if run:
                    seg_ids, s_dec, s_tgt = self._decode_encode_segment(
                        run, src_tok, dst_tok,
                    )
                    all_target_ids.extend(seg_ids)
                    source_parts.append(s_dec)
                    target_parts.append(s_tgt)
                    run = []
                match = self._map_single(tid, src, dst)
                if match.target_ids:
                    all_target_ids.extend(match.target_ids)
                if match.source_str:
                    source_parts.append(match.source_str)
                if match.target_str:
                    target_parts.append(match.target_str)
                overall = _worse(overall, match.confidence)
            else:
                run.append(tid)
        if run:
            seg_ids, s_dec, s_tgt = self._decode_encode_segment(
                run, src_tok, dst_tok,
            )
            all_target_ids.extend(seg_ids)
            source_parts.append(s_dec)
            target_parts.append(s_tgt)

        if len(all_target_ids) != len(token_ids):
            log.debug(
                "map_sequence %s: token count changed %d -> %d for %r",
                src, len(token_ids), len(all_target_ids), "".join(source_parts),
            )

        return Match(
            overall, all_target_ids,
            source_str="".join(source_parts), target_str="".join(target_parts),
        )

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

    def show_info(self) -> None:
        print("Token Matcher: Ouro-1.4B <-> HRM-Text-1B")
        print(f"  Ouro vocab: {len(self.ouro_vocab):>5}  ({len(self.ouro_special):>2} special)")
        print(f"  HRM  vocab: {len(self.hrm_vocab):>5}  ({len(self.hrm_special):>2} special)")
        shared_by_value = set(self.ouro_special.values()) & set(self.hrm_special.values())
        shared_by_id = len(
            set(self.ouro_special.keys()) & set(self.hrm_special.keys())
        )
        print(
            f"  Shared special tokens: "
            f"{len(shared_by_value)} by string value, "
            f"{shared_by_id} by ID"
        )
        print()
