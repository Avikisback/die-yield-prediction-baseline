"""Multi-resolution die yield prediction.

Three resolutions, each matched to a mechanism described in the problem
statement:

    wafer  -- radial and linear process gradients from the chamber
    die    -- 500 parametric measurements plus neighbourhood context
    block  -- 2000 sub-die readings carrying a sparse, clustered defect
              signature

The package keeps ``modeling/`` (the earlier baseline) untouched so that both
pipelines can be scored on identical folds.
"""

__all__ = ["spatial", "parametric", "block", "cache"]
