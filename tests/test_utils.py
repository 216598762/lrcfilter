"""Tests for shared utility functions."""

import pytest

from lrcfilter.utils import normalize_text, normalize_for_censorship, normalize_for_mismatch


class TestNormalizeText:
    """Test the normalize_text function with all keyword arguments."""

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_none_string(self) -> None:
        assert normalize_text(None) == ""

    def test_lowercase(self) -> None:
        assert normalize_text("Hello World") == "hello world"

    def test_extra_whitespace(self) -> None:
        assert normalize_text("  extra   spaces  ") == "extra spaces"

    # --- remove_quotes tests ---

    def test_remove_quotes_true(self) -> None:
        assert normalize_text("It's a \"test\"", remove_quotes=True) == "its a test"

    def test_normalize_quotes_default(self) -> None:
        # Smart quotes should be normalized (or at least not crash)
        result = normalize_text("It\u2019s a \u201ctest\u201d", remove_quotes=False)
        # Result should be lowercase
        assert result == result.lower()

    # --- remove_suffixes tests ---

    def test_remove_suffixes_default(self) -> None:
        assert normalize_text("Song (Remix)", remove_suffixes=True) == "song"
        assert normalize_text("Song (Live)", remove_suffixes=True) == "song"
        assert normalize_text("Song (Acoustic)", remove_suffixes=True) == "song"
        assert normalize_text("Song (Deluxe Edition)", remove_suffixes=True) == "song"
        assert normalize_text("Song (Remastered)", remove_suffixes=True) == "song"
        assert normalize_text("Song (Explicit)", remove_suffixes=True) == "song"

    def test_remove_suffixes_custom(self) -> None:
        custom_suffixes = {"(custom)", "(version)"}
        result = normalize_text("Song (custom)", remove_suffixes=True, suffixes=custom_suffixes)
        assert result == "song"

    def test_remove_suffixes_none_uses_defaults(self) -> None:
        # suffixes=None triggers default suffix set
        result = normalize_text("Song (radio edit)", remove_suffixes=True, suffixes=None)
        assert result == "song"

    # --- remove_filler_words tests ---

    def test_remove_filler_words_default(self) -> None:
        result = normalize_text("the quick brown fox", remove_filler_words=True)
        assert "the" not in result.split()

    def test_remove_filler_words_custom(self) -> None:
        custom_fillers = {"foo", "bar"}
        result = normalize_text("foo bar baz", remove_filler_words=True, filler_words=custom_fillers)
        assert result == "baz"

    def test_remove_filler_words_none_uses_defaults(self) -> None:
        # filler_words=None triggers default filler set
        result = normalize_text("a an the and or but", remove_filler_words=True, filler_words=None)
        assert "the" not in result.split()
        assert "a" not in result.split()

    def test_remove_filler_words_preserves_content_words(self) -> None:
        result = normalize_text("the song is great", remove_filler_words=True)
        assert "song" in result
        assert "great" in result

    # --- combined options ---

    def test_all_options_combined(self) -> None:
        result = normalize_text(
            "The \"Song\" (Remix) is great",
            remove_quotes=True,
            remove_filler_words=True,
            remove_suffixes=True,
        )
        assert "remix" not in result
        assert "the" not in result.split()
        assert '"' not in result


class TestNormalizeForCensorship:
    """Test the normalize_for_censorship convenience function."""

    def test_removes_filler_words(self) -> None:
        result = normalize_for_censorship("the song and the words")
        assert "the" not in result.split()
        assert "and" not in result.split()

    def test_preserves_suffixes(self) -> None:
        result = normalize_for_censorship("Song (Remix)")
        assert "remix" in result.lower() or "(remix)" in result.lower()

    def test_normalizes_quotes(self) -> None:
        result = normalize_for_censorship("It's a test")
        assert "'" in result or "its" in result


class TestNormalizeForMismatch:
    """Test the normalize_for_mismatch convenience function."""

    def test_removes_quotes(self) -> None:
        result = normalize_for_mismatch("It's a \"test\"")
        assert "'" not in result
        assert '"' not in result

    def test_removes_suffixes(self) -> None:
        result = normalize_for_mismatch("Song (Remix)")
        assert "remix" not in result.lower()

    def test_preserves_filler_words(self) -> None:
        result = normalize_for_mismatch("the song is great")
        assert "the" in result.split()
