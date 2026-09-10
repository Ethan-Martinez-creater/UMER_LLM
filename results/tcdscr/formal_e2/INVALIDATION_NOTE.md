# E2 INVALIDATION NOTE

Status: **INVALID**

Reason:
Proxy was trained with `[z_sel ; g_t]` (event representation) but evaluated
with `[z_sel ; h_source]`, violating the frozen E2 proxy contract
(`proxy input = [h_source ; z_sel]`). The training path fed `event_repr`
as the proxy's second 768-D input while the validation path concatenated
the selected representation with the source representation in the reverse
order; the two paths never shared a classifier contract.

The previous readiness values must not be used for scientific claims.

- All results under `results/tcdscr/formal_e2/` are retained unchanged for
  auditability but are INVALID.
- The corrected run lives in `results/tcdscr/formal_e2_corrected/` with
  the unified `proxy.classify([h_source ; z_sel])` contract (soft training
  path and hard validation/baseline path), a true-cosine semantic baseline,
  and epoch-mean loss logging.
- E1, E1 finalization and all earlier stages are unaffected.