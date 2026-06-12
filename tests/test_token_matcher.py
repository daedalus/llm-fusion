"""Tests for llm_fusion.token_matcher."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_fusion.token_matcher import TokenMatcher


@pytest.fixture
def matcher() -> TokenMatcher:
    base = Path(__file__).resolve().parent.parent
    ouro_path = base / "Ouro-1.4B/tokenizer.json"
    hrm_path = base / "HRM-Text-1B/tokenizer.json"
    if not ouro_path.exists() or not hrm_path.exists():
        pytest.skip("model tokenizer files not found")
    return TokenMatcher(str(ouro_path), str(hrm_path))


class TestTokenMatcherInit:
    def test_missing_ouro_tokenizer_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Missing tokenizer"):
            TokenMatcher(
                ouro_tokenizer_path="/nonexistent/ouro.json",
                hrm_tokenizer_path="/nonexistent/hrm.json",
            )

    def test_show_info_runs(self, matcher) -> None:
        matcher.show_info()

    def test_vocab_sizes(self, matcher) -> None:
        assert len(matcher.ouro_vocab) > 0
        assert len(matcher.hrm_vocab) > 0


class TestTokenMatcherMapping:
    def test_ouro_to_hrm_exact_match(self, matcher) -> None:
        m = matcher.ouro_to_hrm(335)
        assert m.confidence == "exact"
        assert 371 in m.target_ids

    def test_ouro_to_hrm_invalid_id(self, matcher) -> None:
        m = matcher.ouro_to_hrm(999999)
        assert m.confidence == "invalid"
        assert m.target_ids == []

    def test_hrm_to_ouro_exact_match(self, matcher) -> None:
        m = matcher.hrm_to_ouro(371)
        assert m.confidence == "exact"
        assert 335 in m.target_ids

    def test_hrm_to_ouro_invalid_id(self, matcher) -> None:
        m = matcher.hrm_to_ouro(999999)
        assert m.confidence == "invalid"
        assert m.target_ids == []

    def test_sequence_mapping_ouro_to_hrm(self, matcher) -> None:
        m = matcher.map_sequence([335, 6783], "ouro")
        assert len(m.target_ids) > 0

    def test_sequence_mapping_hrm_to_ouro(self, matcher) -> None:
        m = matcher.map_sequence([371, 9829], "hrm")
        assert len(m.target_ids) > 0

    def test_format_match_invalid(self, matcher) -> None:
        m = matcher.ouro_to_hrm(999999)
        output = matcher.format_match(m, "OURO", 999999)
        assert "<" in output

    def test_format_match_exact(self, matcher) -> None:
        m = matcher.ouro_to_hrm(335)
        output = matcher.format_match(m, "OURO", 335)
        assert "✓" in output or "~" in output

    def test_format_match_without_src_id(self, matcher) -> None:
        m = matcher.ouro_to_hrm(335)
        output = matcher.format_match(m, "OURO")
        assert "OURO" in output

    def test_format_match_with_note(self, matcher) -> None:
        m = matcher.ouro_to_hrm(0)
        output = matcher.format_match(m, "OURO", 0)
        assert "#" in output or "✓" in output


class TestNormalizeBPE:
    def test_normalize_bpe_replaces_gpt2_space(self, matcher) -> None:
        assert TokenMatcher._normalize_bpe("\u0120hello") == " hello"

    def test_normalize_bpe_no_change(self, matcher) -> None:
        assert TokenMatcher._normalize_bpe("hello") == "hello"

    def test_normalize_bpe_empty(self, matcher) -> None:
        assert TokenMatcher._normalize_bpe("") == ""


class TestNormalizeCandidates:
    def test_starts_with_gpt2_space(self, matcher) -> None:
        candidates = matcher._normalize_candidates("\u0120the")
        assert "the" in candidates
        assert " the" in candidates
        assert "\u0120the" in candidates

    def test_no_gpt2_space(self, matcher) -> None:
        candidates = matcher._normalize_candidates("the")
        assert candidates == ["the"]


class TestRoundTrip:
    def test_ouro_round_trip(self, matcher) -> None:
        for tid in [0, 11, 42, 335, 6783]:
            m = matcher.ouro_to_hrm(tid)
            if m.confidence == "invalid":
                continue
            if m.target_ids:
                back = matcher.hrm_to_ouro(m.target_ids[0])
                assert back.confidence != "invalid"

    def test_hrm_round_trip(self, matcher) -> None:
        for tid in [0, 11, 1738, 22938]:
            m = matcher.hrm_to_ouro(tid)
            if m.confidence == "invalid":
                continue
            if m.target_ids:
                back = matcher.ouro_to_hrm(m.target_ids[0])
                assert back.confidence != "invalid"

    def test_special_tokens_crosswalk(self, matcher) -> None:
        for tid in list(matcher.ouro_special.keys())[:5]:
            m = matcher.ouro_to_hrm(tid)
            assert m.confidence in ("exact", "approx")
            assert "special" in m.note.lower() if m.note else True

    def test_map_sequence_empty_encode(self, matcher) -> None:
        m = matcher.map_sequence([], "ouro")
        assert m.confidence == "approx"

    def test_ouro_to_hrm_special_no_crosswalk(self, matcher) -> None:
        hrm_special_no_ouro = None
        for tid, s in matcher.hrm_special.items():
            if s not in {v for v in matcher.ouro_special.values()}:
                hrm_special_no_ouro = tid
                break
        if hrm_special_no_ouro is not None:
            m = matcher.hrm_to_ouro(hrm_special_no_ouro)
            assert m.confidence == "mismatch"
            assert m.target_ids == []

    def test_map_sequence_with_specials(self, matcher) -> None:
        ouro_ids = list(matcher.ouro_special.keys())[:3]
        m = matcher.map_sequence(ouro_ids, "ouro")
        assert len(m.target_ids) > 0

    def test_map_sequence_hrm_with_specials(self, matcher) -> None:
        hrm_ids = list(matcher.hrm_special.keys())[:3]
        m = matcher.map_sequence(hrm_ids, "hrm")
        assert len(m.target_ids) > 0

    def test_map_sequence_text_preserved(self, matcher) -> None:
        text = "Hello, world!"
        ouro_ids = matcher.ouro_tok.encode(text).ids
        m_oh = matcher.map_sequence(ouro_ids, "ouro")
        m_ho = matcher.map_sequence(m_oh.target_ids, "hrm")
        reconstructed = matcher.ouro_tok.decode(m_ho.target_ids, skip_special_tokens=False)
        assert reconstructed == text

    def test_map_sequence_hrm_text_preserved(self, matcher) -> None:
        text = "Hello, world!"
        hrm_ids = matcher.hrm_tok.encode(text).ids
        m_ho = matcher.map_sequence(hrm_ids, "hrm")
        m_oh = matcher.map_sequence(m_ho.target_ids, "ouro")
        reconstructed = matcher.hrm_tok.decode(m_oh.target_ids, skip_special_tokens=False)
        assert reconstructed == text

    def test_map_sequence_token_count_changed_logs(self, matcher) -> None:
        ouro_ids = matcher.ouro_tok.encode(" the").ids
        m = matcher.map_sequence(ouro_ids, "ouro")
        assert len(m.target_ids) != len(ouro_ids)
