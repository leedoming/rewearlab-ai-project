"""Per-image detection cache shared across bbox-policy experiments.

IMPLEMENTATION_SPEC.md section 19 (Phase 4 - BBox Policy Experiments) MUST:
"모든 policy는 동일한 detection output에 적용한다. Detector를 policy별로
다시 돌려 결과가 달라지는 구조를 피한다." (Every policy must run against
the same detection output; do not re-run the detector per policy, which
would let results differ for reasons unrelated to the policy itself.)

A cache makes this a structural guarantee instead of a convention each
experiment script has to remember: the underlying (expensive) detector is
invoked at most once per image path, and every policy reads the same
cached result.
"""


class DetectionCache:
    """Memoizes `detect(image_path)` by image path."""

    def __init__(self, detect):
        self._detect = detect
        self._cache = {}

    def get(self, image_path):
        key = str(image_path)
        if key not in self._cache:
            self._cache[key] = self._detect(image_path)
        return self._cache[key]

    def __len__(self):
        return len(self._cache)
