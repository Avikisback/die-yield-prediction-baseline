"""Generator-matched fusion for die-yield prediction.

Where ``mrf`` fits a linear head over engineered features, this package writes
down the posterior the data generating process actually implies and estimates
its pieces:

    logit P(fail | die) = logit(prior_i) + logLR_parametric(s_i) + logLR_block(b_i)

``prior_i`` is the wafer's failure rate times the pre-test hazard shape, and the
two evidence terms are the only places the die's own measurements enter.  Every
piece is estimated, none is read from ``config.yaml``; the report checks the
estimates against the generator afterwards.
"""

__all__ = [
    "blocks",
    "channels",
    "hazard",
    "head",
    "waferrate",
]
