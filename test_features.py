import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from features import extract_numeric_features


def test_extract_numeric_features_counts_exclamation_and_caps():
    data = extract_numeric_features(["WOW!!! This is great", "no caps here"])
    assert int(data.loc[0, "exclamation_count"]) == 3
    assert int(data.loc[0, "all_caps_tokens"]) >= 1
    assert int(data.loc[1, "exclamation_count"]) == 0


def test_repeated_phrases_detected():
    data = extract_numeric_features(["Highly recommend this product with a free sample!"])
    assert int(data.loc[0, "repeated_phrases"]) >= 1
