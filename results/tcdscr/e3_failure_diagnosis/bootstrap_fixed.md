# E3 Paired Bootstrap (fixed)

## Previous bug
The E3_TEST_REPORT bootstrap converted the with-replacement resampled event list into a dict keyed by event_id, so events drawn more than once were folded into a single occurrence. Bootstrap multiplicity was lost and the 95% CIs were invalid.

## Corrected implementation
Sampling unit = test event; 10,000 iterations; seed = 3090; each iteration draws n events WITH replacement and every occurrence contributes its predictions (the sampled list is never re-keyed into a unique-event dict). Paired Static vs Dynamic on the same events.

Model predictions and point estimates are unaffected.

## PHEME

Delta Macro-F1 point: +0.00010
95% CI: [-0.00010, +0.00032]

Delta FlipRate point: +0.00069
95% CI: [+0.00040, +0.00101]

## MAWEIBO

Delta Macro-F1 point: +0.00001
95% CI: [-0.00014, +0.00016]

Delta FlipRate point: -0.00002
95% CI: [-0.00024, +0.00021]
