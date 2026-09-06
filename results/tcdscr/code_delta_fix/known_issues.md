# TC-DSCR V2 Code Complete — Known Issues / Implementation Decisions

1. **Unreachable nodes keep the historical -1 depth sentinel.** A node whose
   causal-safe edge set cannot reach the source (rare: parent excluded or
   child-before-parent) gets depth = -1, hence norm_depth = -1/19. This is the
   historical `.pt` behavior; the count is audited per snapshot
   (`unreachable_count`) and in the dataset audit summary.
2. **Depth > 19 is recorded, never clipped** (§9.2): see
   `depth_overflow_count` per snapshot.
3. **Prompt/LLM text is the raw social text** (no extra normalization); the
   frozen cleaning functions (clean_tweet_pheme / clean_text_weibo) apply to
   the §7 semantic pipeline only. The V2 plan does not specify prompt-side
   cleaning; keeping raw text preserves social cues for the LLM, and this
   decision is recorded here rather than invented silently.
4. **Ma-Weibo node text prefers `original_text`, falling back to `text` only
   when the field is absent** — the exact historical adapter behavior verified
   by the D6 parity run (coverage 99.9996%).
5. **Dynamic memory is single-step rolling**: M_{t-1} is the evidence selected
   at the immediately preceding snapshot of the same event; M_t is overwritten
   after each snapshot. SOURCE_ONLY leaves memory empty, so the first dynamic
   snapshot has novelty = 1 everywhere.
6. **Smoke uses fixed lambda_n=0.5, lambda_p=0.1** (mid-grid values, no
   tuning); formal grid search happens in Stage B on validation only.
7. **LLM request cap**: at most 10 requests per dataset (20 total), consumed
   in deterministic (event_id, cutoff) enumeration order; the remaining
   snapshot rows carry prediction=null and are excluded from metrics.
8. **recent_budget tie-break**: equal timestamps keep snapshot order
   (earlier first). The V2 plan fixes tie-breaking only for the structural
   budget; this choice is recorded for determinism.
9. **SOURCE_ONLY prompts** show evidence block `(none)` and
   observed_replies = 0; the source claim is always provided separately and
   never appears as an evidence unit.
10. **MockRumorLLM exists for unit/integration tests only**; the smoke run
    uses the real frozen Qwen3-8B with greedy decoding and thinking disabled.
11. **Training pools**: SOURCE_ONLY snapshots are excluded from encoder
    training (§36); the selector trains on the same dynamic pool with the
    frozen encoder.
12. **Adapter strictness**: timestamp parse failures raise immediately
    (frozen audit: 100% coverage) instead of silently dropping nodes.
