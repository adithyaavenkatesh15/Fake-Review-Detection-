import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from clean_text import clean_text, batch_clean


def test_clean_text_removes_urls_html_and_punctuation():
    source = "Check this out: https://example.com <b>Hi!</b>"
    assert clean_text(source) == "check this out hi"


def test_batch_clean_preserves_order():
    assert batch_clean(["Hello!!", "Test 123"]) == ["hello", "test"]
