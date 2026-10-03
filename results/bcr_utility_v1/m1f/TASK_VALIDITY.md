# BCR-Utility M1-F — Task A: Task-Validity Audit

```text
protocol                    = bcr_v1
relabelled_historical_data  = False
changed_threshold_or_split  = False
```

## 1. What the frozen reader actually predicts

* prompt builder: `project/cr_tser/readers/base_reader.py`
* target: A = RUMOR vs B = NON_RUMOR (source post plus the observed social evidence visible up to the cutoff)
* scoring: teacher-forced candidate log probability p_r(y|C), y in {A=RUMOR, B=NON_RUMOR}

The prompt is fixed text: the source post, the observed social evidence up to the cutoff, and the instruction to answer with a single letter where `A = RUMOR` and `B = NON_RUMOR`. The reader is never asked whether the claim is factually true.

## 2. What `utility` therefore means

```text
u_r(e) = p_r(gold | SRC) - p_r(gold | SRC without e)
gold      = the dataset's frozen rumour/non-rumour label of the event the evidence belongs to
threshold = +-0.05
supervision = I1_atomic only; I2-I5 are non-supervision
```

* positive utility: the evidence unit makes this reader more likely to answer with the dataset's gold rumour/non-rumour label for that event
* negative utility: the evidence unit makes this reader less likely to answer with that gold label
* is verification utility: **False**

**Claim allowed:** reader-specific rumour-label evidence utility.

**Claim forbidden:** fact-checking / truth-verification utility, or any claim that the evidence was shown to be true or false.

## 3. Dataset label semantics

### maweibo

```text
platform            = Sina Weibo
language            = Chinese
task                = event-level rumour vs non-rumour detection
label_semantics     = the event's rumour / non-rumour annotation carried by the frozen Ma-Weibo source (Ma et al., IJCAI 2016)
is_truth_verification = False
```

the label is a rumour annotation, not a verified true/false verdict on the underlying claim.

### pheme

```text
platform            = Twitter
language            = English
task                = event-level rumour vs non-rumour detection
label_semantics     = the event's rumour / non-rumour annotation carried by the frozen PHEME source (Zubiaga et al., PLOS ONE 2016)
is_truth_verification = False
```

rumour != false and non-rumour != true: PHEME annotates whether a claim circulated as an unverified rumour, not whether it is factually true or false.

## 4. Cross-dataset boundary

* language-only shift: **False**
* Ma-Weibo and PHEME differ in platform, language, annotation process and claim distribution; M1-E measured a systematic E3 distribution shift, so the M1-F contrast must not be reported as a language-only shift

## 5. Conclusion

```text
supports_intended_claim = True
validity_blocked        = False
```

utility is defined over the reader's rumour/non-rumour answer agreement, so an evidence-level claim is supported provided it is stated as rumour-label utility and never as truth verification.

Sources re-read for this audit: `project/cr_tser/config/pilot_config.py`, `project/cr_tser/readers/base_reader.py`, `project/cr_tser/readers/sequence_scorer.py`
