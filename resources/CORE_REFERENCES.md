# Core references used by UMER

This ledger intentionally contains only sources directly relevant to UMER's
datasets, architecture, or optimization. Comparison-method searches and
comparison experiments are excluded from the handoff workspace.

| Role | Reference | Used for |
|---|---|---|
| PHEME dataset | Zubiaga et al., *Analysing How People Orient to and Spread Rumours in Social Media by Looking at Conversational Threads*, PLOS ONE, 2016. https://doi.org/10.1371/journal.pone.0150989 | PHEME conversational-event definition and data provenance |
| Ma-Weibo dataset | Ma et al., *Detect Rumors from Microblogs with Recurrent Neural Networks*, IJCAI 2016. https://www.ijcai.org/Proceedings/16/Papers/537.pdf | Ma-Weibo data provenance and event-level rumor task |
| DeBERTa | He et al., *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*, ICLR 2021. https://openreview.net/forum?id=XPZIaotutsD | Disentangled content/position language encoding background |
| DeBERTa-v3 | He et al., *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*, 2023. https://openreview.net/forum?id=sE7-XhLxHA | Final `DeBERTa-v3-large` text encoder |
| Sentence-BERT | Reimers and Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP-IJCNLP 2019. https://doi.org/10.18653/v1/D19-1410 | Semantic-neighborhood motivation; UMER itself retrieves from supplied event features rather than an external SBERT model |
| R-Drop | Liang et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021. https://arxiv.org/abs/2106.14448 | Prediction-consistency background; final UMER uses consistency across two label-preserving event serializations, not two identical-input R-Drop passes |
| SAM | Foret et al., *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021. https://openreview.net/forum?id=6Tm1mposlrM | Sharpness-aware parameter update; UMER adds exact effective-batch/RNG replay for stochastic views |
| LLM cognitive advisor | Hu et al., *Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection*, AAAI 2024. https://doi.org/10.1609/aaai.v38i20.30214 | Motivates using an LLM as a multi-perspective offline advisor rather than the final detector; initial UMER experiment `umer_cognitive_schema_audit_qwen_size_v1` |
| LLM cognitive distillation | Chen et al., *Enhancing text-centric fake news detection via external knowledge distillation from LLMs*, Neural Networks 2025. https://doi.org/10.1016/j.neunet.2025.107377 | Training-only LLM knowledge generation and inference-free distillation background for Direction 1; initial UMER experiment `umer_cognitive_schema_audit_qwen_size_v1` |
| Structured social-context refinement | Zeng et al., *Exploring Large Language Models for Effective Rumor Detection on Social Media*, NAACL 2025. https://doi.org/10.18653/v1/2025.naacl-long.128 | Motivates moderate, propagation-aware context construction rather than feeding all replies directly to an LLM |
| Explanation faithfulness | Madsen et al., *Are self-explanations from Large Language Models faithful?*, Findings of ACL 2024. https://doi.org/10.18653/v1/2024.findings-acl.19 | Requires intervention-based validation before treating generated cognitive rationales as faithful explanations |

Before adding a new model mechanism, append its primary source here and record
the exact UMER experiment ID that tests it. Do not add papers used only as
comparison methods to this file.
