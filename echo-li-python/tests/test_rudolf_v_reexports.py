"""The frontend classes are Rudolf-V's binding, re-exported: one class each, so
a `Frontend` from either package is the one `Stereo.range_priors` takes."""

import echo_li
import rudolf_v


def test_frontend_classes_are_rudolf_vs():
    for name in ["Frontend", "FrontendConfig", "Stereo", "CheckerboardDetector"]:
        assert getattr(echo_li, name) is getattr(rudolf_v, name)
