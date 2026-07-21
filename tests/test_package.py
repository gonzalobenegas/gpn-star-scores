from importlib.metadata import version

import gpn_star_scores


def test_package_version_matches_metadata() -> None:
    assert gpn_star_scores.__version__ == version("gpn-star-scores")
