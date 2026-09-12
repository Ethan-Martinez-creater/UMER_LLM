# TC-DSCR Prohibited Claims

These statements are **not supported** by the current evidence. They must not appear in a paper, abstract, slide or rebuttal. Each entry gives the reason and the acceptable alternative phrasing.

Freeze commit: `c96ccbddbc5aeecfd62efa5e162ad0aff33290c2`.

```json
{
  "prohibited_claim_ids": ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"],
  "n_prohibited": 7
}
```

---

### PC1 — "TC-DSCR Dynamic Memory significantly improves rumor detection." (禁止 1)

**Reason.** Dynamic V1 failed. On the held-out test the delta Macro-F1 was +0.0001037 (PHEME) and +0.0000120 (Ma-Weibo), and both corrected paired bootstrap CIs cross zero. The method is REJECTED as a method (negative finding retained).

**Allowed alternative.** "A point-wise temporal novelty/persistence reranking did not improve rumor detection; we report the measured mechanism (score-scale mismatch and budget masking) as a design finding."

---

### PC2 — "MF-TSR improves classification by better matching the full graph." (禁止 2)

**Reason.** Teacher-fidelity improvement did not transfer consistently: MF-TSR moved Macro-F1 by +0.00293 on PHEME and -0.00286 on Ma-Weibo, and the frozen gate FAILED on both datasets. Distortion reduction is not equivalent to discriminative sufficiency.

**Allowed alternative.** "Set-level refinement driven by a teacher-fidelity objective changed the evidence context substantially but did not reliably improve downstream classification."

---

### PC3 — "MS-TSR finds universally minimal sufficient social evidence." (禁止 3)

**Reason.** Ma-Weibo Qwen transfer failed the frozen gate (-0.00933 Macro-F1, label TRANSFER_FAIL). Sufficiency was defined and verified only against the frozen proxy, and the V3-B diagnosis recommends `MS_TSR_COMPRESSION_ONLY`.

**Allowed alternative.** "MS-TSR finds a low-cost evidence subset that preserves the frozen proxy decision; the resulting context is a compression of that decision, not a universally sufficient context for an arbitrary reader."

---

### PC4 — "Proxy confidence reliably predicts LLM evidence sufficiency." (禁止 4)

**Reason.** The V3-B failure diagnosis flagged `PROXY_READER_MARGIN_MISMATCH`: mean proxy margin retention was 1.2986 (above 1, i.e. the compressed set retained or exceeded the static margin) while the reader delta was -0.00933, and proxy-margin retention vs the correctness transition had Spearman -0.0929 (Ma-Weibo) / +0.0751 (PHEME). No proxy-side signal cleared its fixed threshold.

**Allowed alternative.** "Proxy-level margin did not predict reader-level degradation in this pilot; the mismatch is the finding, not a usable signal."

---

### PC5 — "Context compression eliminates hallucination." (禁止 5)

**Reason.** Only evidence-ID validity was evaluated: whether each cited E# exists in the supplied block (valid citation rate 1.0, unsupported citation rate 0.0). That check does not verify that the cited evidence supports the stated reason and does not cover any other hallucination form.

**Allowed alternative.** "Fabricated evidence-ID references were not observed in this pilot (narrow scope: citation-ID existence only)."

---

### PC6 — "PHEME +1.97pt Qwen improvement is statistically significant." (禁止 6)

**Reason.** The paired bootstrap CI crosses zero: point +0.01974, 95% CI [-0.02192, +0.06144]. The exact McNemar p-value is 0.4296 (17 static-correct/MS-wrong vs 23 static-wrong/MS-correct).

**Allowed alternative.** "The PHEME reader delta was positive but not statistically significant; the label is WEAK_TRANSFER and the overall pilot verdict is PARTIAL."

---

### PC7 — "Ma-Weibo degradation proves context compression is harmful." (禁止 7)

**Reason.** Only one validation reader pilot exists (one model, one prompt); the Ma-Weibo delta -0.00933 is itself inside its confidence interval [-0.04010, +0.02129], so the effect is dataset- and context-dependent rather than established harm.

**Allowed alternative.** "MS-TSR compression crossed the pre-registered non-inferiority boundary on Ma-Weibo in this single validation pilot; the effect is not statistically significant and must not be generalized."

---

## Additional phrasing rules

- Never write "never transfers"; write "does not automatically transfer".
- Never write "MS-TSR is a minimal-sufficient selector"; write "MS-TSR is a compression component".
- Never label any V3-B number as a held-out test result; it is a VALIDATION PILOT.
- Never present the MS-TSR ΔMacro-F1 = 0 as an experimental finding; it is a structural consequence of condition C1 plus static fallback.
