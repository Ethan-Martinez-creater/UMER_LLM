# TC-DSCR V3-B Reader-Transfer Failure Diagnosis

Post-hoc diagnosis over the frozen 600 paired samples / 1200 generations. No new Qwen inference, no resampling, no MS-TSR change.

## Overall Finding

- paired outcome groups: pheme {'CC': 172, 'CW': 17, 'WC': 23, 'WW': 88, 'n': 300, 'net_correction': 6}, maweibo {'CC': 252, 'CW': 13, 'WC': 10, 'WW': 25, 'n': 300, 'net_correction': -3}
- root causes flagged: ['PROXY_READER_MARGIN_MISMATCH', 'DATASET_SPECIFIC_CONTEXT_NEED']
- recommendation: MS_TSR_COMPRESSION_ONLY

## Frozen V3-B Results

| dataset | Static MF1 | MS MF1 | delta | social reduction |
|---|---:|---:|---:|---:|
| pheme | 0.6278 | 0.6476 | 0.01974 | 0.6870 |
| maweibo | 0.8825 | 0.8732 | -0.00933 | 0.8532 |

## 1. Paired Outcome Groups

```json
{
 "pheme": {
  "CC": 172,
  "CW": 17,
  "WC": 23,
  "WW": 88,
  "n": 300,
  "net_correction": 6
 },
 "maweibo": {
  "CC": 252,
  "CW": 13,
  "WC": 10,
  "WW": 25,
  "n": 300,
  "net_correction": -3
 }
}
```

## 2. Compression Severity

```json
{
 "pheme": {
  "CC": {
   "n": 172,
   "token_reduction_ratio": {
    "n": 138,
    "mean": 0.6779498238855195,
    "median": 0.8765690376569037,
    "p25": 0.5257142857142857,
    "p50": 0.8765690376569037,
    "p75": 0.9274106175514626,
    "p90": 0.9540481400437637
   },
   "evidence_count_reduction_ratio": {
    "n": 138,
    "mean": 0.6599100417674583,
    "median": 0.8571428571428572,
    "p25": 0.5,
    "p50": 0.8571428571428572,
    "p75": 0.9166666666666666,
    "p90": 0.9285714285714286
   },
   "ms_selected_count": {
    "n": 172,
    "mean": 1.0755813953488371,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 1
   },
   "static_selected_count": {
    "n": 172,
    "mean": 5.813953488372093,
    "median": 5,
    "p25": 1,
    "p50": 5,
    "p75": 11,
    "p90": 13
   },
   "ms_social_tokens": {
    "n": 172,
    "mean": 71.15697674418605,
    "median": 57,
    "p25": 36,
    "p50": 57,
    "p75": 71,
    "p90": 85
   },
   "disagreement_rate": 0.0
  },
  "CW": {
   "n": 17,
   "token_reduction_ratio": {
    "n": 17,
    "mean": 0.8834619597677036,
    "median": 0.9152173913043479,
    "p25": 0.8764705882352941,
    "p50": 0.9152173913043479,
    "p75": 0.9501039501039501,
    "p90": 0.9555555555555556
   },
   "evidence_count_reduction_ratio": {
    "n": 17,
    "mean": 0.8465679745091511,
    "median": 0.9,
    "p25": 0.8333333333333334,
    "p50": 0.9,
    "p75": 0.9230769230769231,
    "p90": 0.9285714285714286
   },
   "ms_selected_count": {
    "n": 17,
    "mean": 1.0,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 1
   },
   "static_selected_count": {
    "n": 17,
    "mean": 9.294117647058824,
    "median": 10,
    "p25": 6,
    "p50": 10,
    "p75": 13,
    "p90": 14
   },
   "ms_social_tokens": {
    "n": 17,
    "mean": 56.588235294117645,
    "median": 52,
    "p25": 45,
    "p50": 52,
    "p75": 66,
    "p90": 78
   },
   "disagreement_rate": 1.0
  },
  "WC": {
   "n": 23,
   "token_reduction_ratio": {
    "n": 23,
    "mean": 0.8220657278765889,
    "median": 0.8758169934640523,
    "p25": 0.7178571428571429,
    "p50": 0.8758169934640523,
    "p75": 0.9369918699186992,
    "p90": 0.9503171247357294
   },
   "evidence_count_reduction_ratio": {
    "n": 23,
    "mean": 0.8053927352840395,
    "median": 0.8333333333333334,
    "p25": 0.75,
    "p50": 0.8333333333333334,
    "p75": 0.9166666666666666,
    "p90": 0.9333333333333333
   },
   "ms_selected_count": {
    "n": 23,
    "mean": 1.0,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 1
   },
   "static_selected_count": {
    "n": 23,
    "mean": 7.739130434782608,
    "median": 6,
    "p25": 4,
    "p50": 6,
    "p75": 12,
    "p90": 15
   },
   "ms_social_tokens": {
    "n": 23,
    "mean": 66.04347826086956,
    "median": 62,
    "p25": 51,
    "p50": 62,
    "p75": 79,
    "p90": 87
   },
   "disagreement_rate": 1.0
  },
  "WW": {
   "n": 88,
   "token_reduction_ratio": {
    "n": 74,
    "mean": 0.616774713392592,
    "median": 0.8204225352112676,
    "p25": 0.0,
    "p50": 0.8204225352112676,
    "p75": 0.912906610703043,
    "p90": 0.9480122324159022
   },
   "evidence_count_reduction_ratio": {
    "n": 74,
    "mean": 0.6031405531405529,
    "median": 0.8,
    "p25": 0.0,
    "p50": 0.8,
    "p75": 0.9090909090909091,
    "p90": 0.9285714285714286
   },
   "ms_selected_count": {
    "n": 88,
    "mean": 1.4772727272727273,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 3
   },
   "static_selected_count": {
    "n": 88,
    "mean": 5.443181818181818,
    "median": 4,
    "p25": 1,
    "p50": 4,
    "p75": 10,
    "p90": 13
   },
   "ms_social_tokens": {
    "n": 88,
    "mean": 106.43181818181819,
    "median": 65,
    "p25": 47,
    "p50": 65,
    "p75": 86,
    "p90": 235
   },
   "disagreement_rate": 0.0
  },
  "cw_vs_cc_wc": {
   "cw_mean_reduction": 0.8834619597677035,
   "cc_wc_mean_reduction": 0.6985378101699578,
   "cw_mean_ms_units": 1.0,
   "cc_wc_mean_ms_units": 1.0666666666666667
  }
 },
 "maweibo": {
  "CC": {
   "n": 252,
   "token_reduction_ratio": {
    "n": 238,
    "mean": 0.8479306446003234,
    "median": 0.9488977955911824,
    "p25": 0.8702213279678068,
    "p50": 0.9488977955911824,
    "p75": 0.9653882132834425,
    "p90": 0.9716520039100685
   },
   "evidence_count_reduction_ratio": {
    "n": 238,
    "mean": 0.8184361889299483,
    "median": 0.8888888888888888,
    "p25": 0.8571428571428572,
    "p50": 0.8888888888888888,
    "p75": 0.9230769230769231,
    "p90": 0.9565217391304348
   },
   "ms_selected_count": {
    "n": 252,
    "mean": 1.1031746031746033,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 1
   },
   "static_selected_count": {
    "n": 252,
    "mean": 9.892857142857142,
    "median": 9,
    "p25": 7,
    "p50": 9,
    "p75": 12,
    "p90": 21
   },
   "ms_social_tokens": {
    "n": 252,
    "mean": 82.55555555555556,
    "median": 44,
    "p25": 32,
    "p50": 44,
    "p75": 106,
    "p90": 132
   },
   "disagreement_rate": 0.0
  },
  "CW": {
   "n": 13,
   "token_reduction_ratio": {
    "n": 13,
    "mean": 0.8909096300538509,
    "median": 0.888454011741683,
    "p25": 0.8678861788617886,
    "p50": 0.888454011741683,
    "p75": 0.9673802242609582,
    "p90": 0.9715736040609138
   },
   "evidence_count_reduction_ratio": {
    "n": 13,
    "mean": 0.863770852470543,
    "median": 0.875,
    "p25": 0.875,
    "p50": 0.875,
    "p75": 0.8888888888888888,
    "p90": 0.9411764705882353
   },
   "ms_selected_count": {
    "n": 13,
    "mean": 1.0,
    "median": 1,
    "p25": 1,
    "p50": 1,
    "p75": 1,
    "p90": 1
   },
   "static_selected_count": {
    "n": 13,
    "mean": 9.153846153846153,
    "median": 8,
    "p25": 8,
    "p50": 8,
    "p75": 9,
    "p90": 17
   },
   "ms_social_tokens": {
    "n": 13,
    "mean": 74.76923076923077,
    "median": 41,
    "p25": 32,
    "p50": 41,
    "p75": 129,
    "p90": 137
   },
   "disagreement_rate": 1.0
  },
```

## 3. Evidence Count Threshold

```json
{
 "pheme": {
  "ms_selected_count_bins": {
   "0": {
    "n": 53,
    "static_accuracy": 0.6981132075471698,
    "ms_accuracy": 0.6981132075471698,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 1.0
   },
   "1": {
    "n": 231,
    "static_accuracy": 0.6320346320346321,
    "ms_accuracy": 0.658008658008658,
    "delta_accuracy": 0.025974025974025983,
    "disagreement_rate": 0.17316017316017315,
    "cw_rate": 0.0735930735930736,
    "wc_rate": 0.09956709956709957,
    "mean_token_reduction": 0.7278128551708487
   },
   "2": {
    "n": 1,
    "static_accuracy": 0.0,
    "ms_accuracy": 0.0,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "3-4": {
    "n": 3,
    "static_accuracy": 0.0,
    "ms_accuracy": 0.0,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "5-8": {
    "n": 4,
    "static_accuracy": 0.5,
    "ms_accuracy": 0.5,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "9+": {
    "n": 8,
    "static_accuracy": 0.5,
    "ms_accuracy": 0.5,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   }
  },
  "selected_count_delta_bins": {
   "0": {
    "n": 94,
    "static_accuracy": 0.648936170212766,
    "ms_accuracy": 0.648936170212766,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "1-2": {
    "n": 44,
    "static_accuracy": 0.5909090909090909,
    "ms_accuracy": 0.6363636363636364,
    "cw_rate": 0.06818181818181818,
    "wc_rate": 0.11363636363636363,
    "mean_token_reduction": 0.6325269719676405
   },
   "3-5": {
    "n": 46,
    "static_accuracy": 0.45652173913043476,
    "ms_accuracy": 0.5652173913043478,
    "cw_rate": 0.043478260869565216,
    "wc_rate": 0.15217391304347827,
    "mean_token_reduction": 0.8210858184619589
   },
   "6-10": {
    "n": 54,
    "static_accuracy": 0.7222222222222222,
    "ms_accuracy": 0.7037037037037037,
    "cw_rate": 0.1111111111111111,
    "wc_rate": 0.09259259259259259,
    "mean_token_reduction": 0.9069185971346271
   },
   "11+": {
    "n": 62,
    "static_accuracy": 0.6774193548387096,
    "ms_accuracy": 0.6774193548387096,
    "cw_rate": 0.0967741935483871,
    "wc_rate": 0.0967741935483871,
    "mean_token_reduction": 0.9443553368285463
   }
  }
 },
 "maweibo": {
  "ms_selected_count_bins": {
   "0": {
    "n": 22,
    "static_accuracy": 1.0,
    "ms_accuracy": 1.0,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 1.0
   },
   "1": {
    "n": 271,
    "static_accuracy": 0.8745387453874539,
    "ms_accuracy": 0.8634686346863468,
    "delta_accuracy": -0.011070110701107083,
    "disagreement_rate": 0.08487084870848709,
    "cw_rate": 0.04797047970479705,
    "wc_rate": 0.03690036900369004,
    "mean_token_reduction": 0.8709243705705492
   },
   "2": {
    "n": 0,
    "static_accuracy": null,
    "ms_accuracy": null,
    "delta_accuracy": null,
    "disagreement_rate": null,
    "cw_rate": null,
    "wc_rate": null,
    "mean_token_reduction": null
   },
   "3-4": {
    "n": 0,
    "static_accuracy": null,
    "ms_accuracy": null,
    "delta_accuracy": null,
    "disagreement_rate": null,
    "cw_rate": null,
    "wc_rate": null,
    "mean_token_reduction": null
   },
   "5-8": {
    "n": 3,
    "static_accuracy": 0.6666666666666666,
    "ms_accuracy": 0.6666666666666666,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "9+": {
    "n": 4,
    "static_accuracy": 1.0,
    "ms_accuracy": 1.0,
    "delta_accuracy": 0.0,
    "disagreement_rate": 0.0,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   }
  },
  "selected_count_delta_bins": {
   "0": {
    "n": 31,
    "static_accuracy": 0.967741935483871,
    "ms_accuracy": 0.967741935483871,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.0
   },
   "1-2": {
    "n": 21,
    "static_accuracy": 0.8095238095238095,
    "ms_accuracy": 0.7619047619047619,
    "cw_rate": 0.047619047619047616,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.6182485986544243
   },
   "3-5": {
    "n": 22,
    "static_accuracy": 0.9090909090909091,
    "ms_accuracy": 0.8636363636363636,
    "cw_rate": 0.045454545454545456,
    "wc_rate": 0.0,
    "mean_token_reduction": 0.8595603963902453
   },
   "6-10": {
    "n": 147,
    "static_accuracy": 0.8707482993197279,
    "ms_accuracy": 0.8571428571428571,
    "cw_rate": 0.061224489795918366,
    "wc_rate": 0.047619047619047616,
    "mean_token_reduction": 0.9269871886076786
   },
   "11+": {
    "n": 79,
    "static_accuracy": 0.8860759493670886,
    "ms_accuracy": 0.8987341772151899,
    "cw_rate": 0.02531645569620253,
    "wc_rate": 0.0379746835443038,
    "mean_token_reduction": 0.960251119075464
   }
  }
 }
}
```

## 4. What Evidence Was Removed?

```json
{
 "pheme": {
  "CC": {
   "removed": {
    "n": 819,
    "mean_utility": 1.4636730964981266,
    "mean_relevance": 0.4652135896146058,
    "mean_depth": 1.4493284493284493,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1473.3675213675215,
    "mean_reply_tokens": 25.36141636141636,
    "mean_parent_tokens": 30.463980463980462,
    "mean_pair_tokens": 76.86568986568986,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7619047619047619,
    "is_memory_previous_rate": 0.031746031746031744,
    "depth_group_rates": {
     "depth1": 0.7924297924297924,
     "depth2": 0.09035409035409035,
     "depth>=3": 0.11721611721611722
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.29914529914529914,
     "middle_50": 0.5018315018315018,
     "newest_quartile": 0.199023199023199
    }
   },
   "retained": {
    "n": 181,
    "mean_utility": 1.7710933078759612,
    "mean_relevance": 0.409337857403798,
    "mean_depth": 1.2320441988950277,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1145.0883977900553,
    "mean_reply_tokens": 21.558011049723756,
    "mean_parent_tokens": 28.281767955801104,
    "mean_pair_tokens": 70.89502762430939,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.6850828729281768,
    "is_memory_previous_rate": 0.5580110497237569,
    "depth_group_rates": {
     "depth1": 0.7679558011049724,
     "depth2": 0.13259668508287292,
     "depth>=3": 0.09944751381215469
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.4696132596685083,
     "middle_50": 0.40331491712707185,
     "newest_quartile": 0.1270718232044199
    }
   }
  },
  "CW": {
   "removed": {
    "n": 143,
    "mean_utility": 0.3122578250778305,
    "mean_relevance": 0.4697043508080604,
    "mean_depth": 1.5314685314685315,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1415.6363636363637,
    "mean_reply_tokens": 27.706293706293707,
    "mean_parent_tokens": 33.37762237762238,
    "mean_pair_tokens": 82.32167832167832,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7482517482517482,
    "is_memory_previous_rate": 0.02097902097902098,
    "depth_group_rates": {
     "depth1": 0.8111888111888111,
     "depth2": 0.09090909090909091,
     "depth>=3": 0.0979020979020979
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3076923076923077,
     "middle_50": 0.5314685314685315,
     "newest_quartile": 0.16083916083916083
    }
   },
   "retained": {
    "n": 15,
    "mean_utility": 2.71226760049661,
    "mean_relevance": 0.3925844188502049,
    "mean_depth": 0.6666666666666666,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 903.5333333333333,
    "mean_reply_tokens": 14.933333333333334,
    "mean_parent_tokens": 23.8,
    "mean_pair_tokens": 60.46666666666667,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.6,
    "is_memory_previous_rate": 0.7333333333333333,
    "depth_group_rates": {
     "depth1": 0.8666666666666667,
     "depth2": 0.06666666666666667,
     "depth>=3": 0.06666666666666667
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3333333333333333,
     "middle_50": 0.5333333333333333,
     "newest_quartile": 0.13333333333333333
    }
   }
  },
  "WC": {
   "removed": {
    "n": 155,
    "mean_utility": 4.86858468834431,
    "mean_relevance": 0.4618390899398304,
    "mean_depth": 1.5161290322580645,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1147.3612903225805,
    "mean_reply_tokens": 27.27741935483871,
    "mean_parent_tokens": 30.26451612903226,
    "mean_pair_tokens": 78.3741935483871,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7870967741935484,
    "is_memory_previous_rate": 0.04516129032258064,
    "depth_group_rates": {
     "depth1": 0.7870967741935484,
     "depth2": 0.09032258064516129,
     "depth>=3": 0.12258064516129032
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3032258064516129,
     "middle_50": 0.4838709677419355,
     "newest_quartile": 0.2129032258064516
    }
   },
   "retained": {
    "n": 23,
    "mean_utility": 4.053725641544746,
    "mean_relevance": 0.4333516156811048,
    "mean_depth": 1.3043478260869565,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1311.4347826086957,
    "mean_reply_tokens": 19.565217391304348,
    "mean_parent_tokens": 30.130434782608695,
    "mean_pair_tokens": 70.73913043478261,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8260869565217391,
    "is_memory_previous_rate": 0.5217391304347826,
    "depth_group_rates": {
     "depth1": 0.8695652173913043,
     "depth2": 0.08695652173913043,
     "depth>=3": 0.043478260869565216
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.391304347826087,
     "middle_50": 0.5217391304347826,
     "newest_quartile": 0.08695652173913043
    }
   }
  },
  "WW": {
   "removed": {
    "n": 351,
    "mean_utility": -0.43635027436406865,
    "mean_relevance": 0.4343022478327168,
    "mean_depth": 1.5042735042735043,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1864.2535612535612,
    "mean_reply_tokens": 25.655270655270655,
    "mean_parent_tokens": 33.21367521367522,
    "mean_pair_tokens": 79.92307692307692,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7834757834757835,
    "is_memory_previous_rate": 0.05128205128205128,
    "depth_group_rates": {
     "depth1": 0.7834757834757835,
     "depth2": 0.09401709401709402,
     "depth>=3": 0.1225071225071225
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.2962962962962963,
     "middle_50": 0.5242165242165242,
     "newest_quartile": 0.1794871794871795
    }
   },
   "retained": {
    "n": 128,
    "mean_utility": 1.1043347072263714,
    "mean_relevance": 0.39700909548334984,
    "mean_depth": 1.4296875,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1356.8046875,
    "mean_reply_tokens": 24.0234375,
    "mean_parent_tokens": 31.3359375,
    "mean_pair_tokens": 76.375,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.734375,
    "is_memory_previous_rate": 0.6953125,
    "depth_group_rates
```

## 5. What Evidence Did Qwen Actually Cite?

```json
{
 "pheme": {
  "CC": {
   "n": 172,
   "static_cited_removed_rate": 0.6752640173835821,
   "static_cited_removed_count": 619,
   "static_cited_total": 776,
   "cited": {
    "n": 776,
    "mean_utility": 1.1363396043049114,
    "mean_relevance": 0.4806815788519994,
    "mean_depth": 1.3028350515463918,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1288.2242268041236,
    "mean_reply_tokens": 24.46778350515464,
    "mean_parent_tokens": 30.905927835051546,
    "mean_pair_tokens": 76.33376288659794,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8195876288659794,
    "is_memory_previous_rate": 0.13659793814432988,
    "depth_group_rates": {
     "depth1": 0.8350515463917526,
     "depth2": 0.08247422680412371,
     "depth>=3": 0.08247422680412371
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3711340206185567,
     "middle_50": 0.47036082474226804,
     "newest_quartile": 0.15850515463917525
    }
   },
   "uncited": {
    "n": 224,
    "mean_utility": 2.84605634739689,
    "mean_relevance": 0.3664784682825849,
    "mean_depth": 1.78125,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1849.4955357142858,
    "mean_reply_tokens": 25.383928571428573,
    "mean_parent_tokens": 27.169642857142858,
    "mean_pair_tokens": 73.88392857142857,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.5,
    "is_memory_previous_rate": 0.09375,
    "depth_group_rates": {
     "depth1": 0.625,
     "depth2": 0.15178571428571427,
     "depth>=3": 0.22321428571428573
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.1875,
     "middle_50": 0.53125,
     "newest_quartile": 0.28125
    }
   }
  },
  "CW": {
   "n": 17,
   "static_cited_removed_rate": 0.855162321338792,
   "static_cited_removed_count": 109,
   "static_cited_total": 119,
   "cited": {
    "n": 119,
    "mean_utility": -0.2838313697892077,
    "mean_relevance": 0.4854315888866016,
    "mean_depth": 1.5630252100840336,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1590.1596638655462,
    "mean_reply_tokens": 27.201680672268907,
    "mean_parent_tokens": 35.30252100840336,
    "mean_pair_tokens": 83.66386554621849,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8067226890756303,
    "is_memory_previous_rate": 0.07563025210084033,
    "depth_group_rates": {
     "depth1": 0.8571428571428571,
     "depth2": 0.06722689075630252,
     "depth>=3": 0.07563025210084033
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.35294117647058826,
     "middle_50": 0.5210084033613446,
     "newest_quartile": 0.12605042016806722
    }
   },
   "uncited": {
    "n": 39,
    "mean_utility": 3.0541747691921697,
    "mean_relevance": 0.3920545992512848,
    "mean_depth": 1.1025641025641026,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 686.1538461538462,
    "mean_reply_tokens": 24.333333333333332,
    "mean_parent_tokens": 23.82051282051282,
    "mean_pair_tokens": 69.82051282051282,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.5128205128205128,
    "is_memory_previous_rate": 0.1282051282051282,
    "depth_group_rates": {
     "depth1": 0.6923076923076923,
     "depth2": 0.15384615384615385,
     "depth>=3": 0.15384615384615385
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.1794871794871795,
     "middle_50": 0.5641025641025641,
     "newest_quartile": 0.2564102564102564
    }
   }
  },
  "WC": {
   "n": 23,
   "static_cited_removed_rate": 0.8399526198439243,
   "static_cited_removed_count": 112,
   "static_cited_total": 129,
   "cited": {
    "n": 129,
    "mean_utility": 4.025768965165051,
    "mean_relevance": 0.45252129252047174,
    "mean_depth": 1.317829457364341,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 964.1162790697674,
    "mean_reply_tokens": 27.170542635658915,
    "mean_parent_tokens": 31.26356589147287,
    "mean_pair_tokens": 79.27906976744185,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8682170542635659,
    "is_memory_previous_rate": 0.10852713178294573,
    "depth_group_rates": {
     "depth1": 0.8682170542635659,
     "depth2": 0.06976744186046512,
     "depth>=3": 0.06201550387596899
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.35658914728682173,
     "middle_50": 0.4806201550387597,
     "newest_quartile": 0.16279069767441862
    }
   },
   "uncited": {
    "n": 49,
    "mean_utility": 6.704941223318479,
    "mean_relevance": 0.47299794624894415,
    "mean_depth": 1.9387755102040816,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1706.795918367347,
    "mean_reply_tokens": 23.93877551020408,
    "mean_parent_tokens": 27.571428571428573,
    "mean_pair_tokens": 72.40816326530613,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.5918367346938775,
    "is_memory_previous_rate": 0.10204081632653061,
    "depth_group_rates": {
     "depth1": 0.6122448979591837,
     "depth2": 0.14285714285714285,
     "depth>=3": 0.24489795918367346
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.20408163265306123,
     "middle_50": 0.5102040816326531,
     "newest_quartile": 0.2857142857142857
    }
   }
  },
  "WW": {
   "n": 88,
   "static_cited_removed_rate": 0.5750349875349875,
   "static_cited_removed_count": 269,
   "static_cited_total": 374,
   "cited": {
    "n": 374,
    "mean_utility": 0.41780333823579197,
    "mean_relevance": 0.457730549198382,
    "mean_depth": 1.3101604278074865,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1838.6122994652405,
    "mean_reply_tokens": 25.02139037433155,
    "mean_parent_tokens": 33.32887700534759,
    "mean_pair_tokens": 79.41711229946524,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.839572192513369,
    "is_memory_previous_rate": 0.24598930481283424,
    "depth_group_rates": {
     "depth1": 0.8475935828877005,
     "depth2": 0.06684491978609626,
     "depth>=3": 0.0855614973262032
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3877005347593583,
     "middle_50": 0.45187165775401067,
     "newest_quartile": 0.1604278
```

## 6. Reader-Sensitive Evidence Loss

```json
{
 "pheme": {
  "per_group": {
   "CC": {
    "n": 172,
    "loss_rate": 0.6752640173835821,
    "mean_loss_count": 3.5988372093023258,
    "samples_with_loss": 111
   },
   "CW": {
    "n": 17,
    "loss_rate": 0.855162321338792,
    "mean_loss_count": 6.411764705882353,
    "samples_with_loss": 16
   },
   "WC": {
    "n": 23,
    "loss_rate": 0.8399526198439243,
    "mean_loss_count": 4.869565217391305,
    "samples_with_loss": 23
   },
   "WW": {
    "n": 88,
    "loss_rate": 0.5750349875349875,
    "mean_loss_count": 3.0568181818181817,
    "samples_with_loss": 51
   }
  },
  "cw_2x2": {
   "a_loss_cw": 16,
   "b_no_loss_cw": 1,
   "c_loss_non_cw": 185,
   "d_no_loss_non_cw": 98
  },
  "odds_ratio": 8.475675675675676,
  "fisher_p": 0.014760996407054413,
  "n": 300
 },
 "maweibo": {
  "per_group": {
   "CC": {
    "n": 252,
    "loss_rate": 0.8380884147727096,
    "mean_loss_count": 7.003968253968254,
    "samples_with_loss": 217
   },
   "CW": {
    "n": 13,
    "loss_rate": 0.9100050276520864,
    "mean_loss_count": 7.3076923076923075,
    "samples_with_loss": 13
   },
   "WC": {
    "n": 10,
    "loss_rate": 0.9838888888888888,
    "mean_loss_count": 7.7,
    "samples_with_loss": 10
   },
   "WW": {
    "n": 25,
    "loss_rate": 0.8572874458874459,
    "mean_loss_count": 6.4,
    "samples_with_loss": 24
   }
  },
  "cw_2x2": {
   "a_loss_cw": 13,
   "b_no_loss_cw": 0,
   "c_loss_non_cw": 251,
   "d_no_loss_non_cw": 36
  },
  "odds_ratio": null,
  "fisher_p": 0.3781149469199706,
  "n": 300
 }
}
```

## 7. Proxy Margin vs Qwen Transfer

```json
{
 "pheme": {
  "bins": {
   "<0.80": {
    "n": 0,
    "cw_rate": null,
    "wc_rate": null,
    "disagreement_rate": null,
    "mean_confidence_delta": null
   },
   "0.80-0.90": {
    "n": 2,
    "cw_rate": 0.5,
    "wc_rate": 0.0,
    "disagreement_rate": 0.5,
    "mean_confidence_delta": -0.425
   },
   "0.90-0.95": {
    "n": 5,
    "cw_rate": 0.0,
    "wc_rate": 0.4,
    "disagreement_rate": 0.4,
    "mean_confidence_delta": -0.019999999999999997
   },
   "0.95-1.00": {
    "n": 111,
    "cw_rate": 0.02702702702702703,
    "wc_rate": 0.018018018018018018,
    "disagreement_rate": 0.04504504504504504,
    "mean_confidence_delta": -0.021171171171171167
   },
   ">1.00": {
    "n": 182,
    "cw_rate": 0.07142857142857142,
    "wc_rate": 0.1043956043956044,
    "disagreement_rate": 0.17582417582417584,
    "mean_confidence_delta": -0.07857142857142857
   }
  },
  "n_compressed": 236,
  "n_static_margin_nonpositive": 0,
  "spearman_retention_confidence_delta": -0.08569219292038606,
  "spearman_retention_transition": 0.0750769778303865
 },
 "maweibo": {
  "bins": {
   "<0.80": {
    "n": 0,
    "cw_rate": null,
    "wc_rate": null,
    "disagreement_rate": null,
    "mean_confidence_delta": null
   },
   "0.80-0.90": {
    "n": 15,
    "cw_rate": 0.0,
    "wc_rate": 0.13333333333333333,
    "disagreement_rate": 0.13333333333333333,
    "mean_confidence_delta": -0.0066666666666666576
   },
   "0.90-0.95": {
    "n": 16,
    "cw_rate": 0.0625,
    "wc_rate": 0.0,
    "disagreement_rate": 0.0625,
    "mean_confidence_delta": -0.05312499999999998
   },
   "0.95-1.00": {
    "n": 47,
    "cw_rate": 0.0,
    "wc_rate": 0.0,
    "disagreement_rate": 0.0,
    "mean_confidence_delta": -0.03085106382978723
   },
   ">1.00": {
    "n": 222,
    "cw_rate": 0.05405405405405406,
    "wc_rate": 0.036036036036036036,
    "disagreement_rate": 0.09009009009009009,
    "mean_confidence_delta": -0.06576576576576579
   }
  },
  "n_compressed": 279,
  "n_static_margin_nonpositive": 0,
  "spearman_retention_confidence_delta": 0.02057297452754712,
  "spearman_retention_transition": -0.0928829726654661
 }
}
```

## 8. Confidence Transition

```json
{
 "pheme": {
  "per_group": {
   "CC": {
    "n": 172,
    "mean_static_confidence": 0.6119186046511631,
    "mean_ms_confidence": 0.550581395348837,
    "mean_confidence_delta": -0.061337209302325585
   },
   "CW": {
    "n": 17,
    "mean_static_confidence": 0.8058823529411764,
    "mean_ms_confidence": 0.5882352941176471,
    "mean_confidence_delta": -0.2176470588235294
   },
   "WC": {
    "n": 23,
    "mean_static_confidence": 0.6782608695652174,
    "mean_ms_confidence": 0.6630434782608696,
    "mean_confidence_delta": -0.015217391304347806
   },
   "WW": {
    "n": 88,
    "mean_static_confidence": 0.5676136363636364,
    "mean_ms_confidence": 0.5335227272727274,
    "mean_confidence_delta": -0.03409090909090908
   }
  },
  "correctness_cells": {
   "static_correct__ms_correct": {
    "n": 172,
    "mean_static_confidence": 0.6119186046511631,
    "mean_ms_confidence": 0.550581395348837
   },
   "static_correct__ms_wrong": {
    "n": 17,
    "mean_static_confidence": 0.8058823529411764,
    "mean_ms_confidence": 0.5882352941176471
   },
   "static_wrong__ms_correct": {
    "n": 23,
    "mean_static_confidence": 0.6782608695652174,
    "mean_ms_confidence": 0.6630434782608696
   },
   "static_wrong__ms_wrong": {
    "n": 88,
    "mean_static_confidence": 0.5676136363636364,
    "mean_ms_confidence": 0.5335227272727274
   }
  }
 },
 "maweibo": {
  "per_group": {
   "CC": {
    "n": 252,
    "mean_static_confidence": 0.7281746031746026,
    "mean_ms_confidence": 0.6561507936507938,
    "mean_confidence_delta": -0.07202380952380955
   },
   "CW": {
    "n": 13,
    "mean_static_confidence": 0.6692307692307692,
    "mean_ms_confidence": 0.7115384615384616,
    "mean_confidence_delta": 0.04230769230769231
   },
   "WC": {
    "n": 10,
    "mean_static_confidence": 0.575,
    "mean_ms_confidence": 0.7999999999999999,
    "mean_confidence_delta": 0.225
   },
   "WW": {
    "n": 25,
    "mean_static_confidence": 0.738,
    "mean_ms_confidence": 0.6719999999999999,
    "mean_confidence_delta": -0.066
   }
  },
  "correctness_cells": {
   "static_correct__ms_correct": {
    "n": 252,
    "mean_static_confidence": 0.7281746031746026,
    "mean_ms_confidence": 0.6561507936507938
   },
   "static_correct__ms_wrong": {
    "n": 13,
    "mean_static_confidence": 0.6692307692307692,
    "mean_ms_confidence": 0.7115384615384616
   },
   "static_wrong__ms_correct": {
    "n": 10,
    "mean_static_confidence": 0.575,
    "mean_ms_confidence": 0.7999999999999999
   },
   "static_wrong__ms_wrong": {
    "n": 25,
    "mean_static_confidence": 0.738,
    "mean_ms_confidence": 0.6719999999999999
   }
  }
 }
}
```

## 9. Label Asymmetry

```json
{
 "pheme": {
  "RUMOR": {
   "n": 112,
   "static_accuracy": 0.7410714285714286,
   "ms_accuracy": 0.7589285714285714,
   "delta_accuracy": 0.017857142857142794,
   "cw": 6,
   "wc": 8,
   "mean_token_reduction": 0.669746063552588,
   "mean_evidence_count_reduction": 0.6527730943093892,
   "mean_reader_evidence_loss_rate": 0.6636766159372539,
   "confusion_transitions": {
    "RUMOR->NON_RUMOR": 6,
    "NON_RUMOR->RUMOR": 8
   }
  },
  "NON_RUMOR": {
   "n": 188,
   "static_accuracy": 0.5638297872340425,
   "ms_accuracy": 0.5851063829787234,
   "delta_accuracy": 0.021276595744680882,
   "cw": 11,
   "wc": 15,
   "mean_token_reduction": 0.697269870699511,
   "mean_evidence_count_reduction": 0.6788291411988355,
   "mean_reader_evidence_loss_rate": 0.6785448816303243,
   "confusion_transitions": {
    "RUMOR->NON_RUMOR": 15,
    "NON_RUMOR->RUMOR": 11
   }
  }
 },
 "maweibo": {
  "RUMOR": {
   "n": 162,
   "static_accuracy": 0.8950617283950617,
   "ms_accuracy": 0.8395061728395061,
   "delta_accuracy": -0.05555555555555558,
   "cw": 10,
   "wc": 1,
   "mean_token_reduction": 0.8084511202305013,
   "mean_evidence_count_reduction": 0.7788561043544081,
   "mean_reader_evidence_loss_rate": 0.7816330882624997,
   "confusion_transitions": {
    "RUMOR->NON_RUMOR": 10,
    "NON_RUMOR->RUMOR": 1
   }
  },
  "NON_RUMOR": {
   "n": 138,
   "static_accuracy": 0.8695652173913043,
   "ms_accuracy": 0.9130434782608695,
   "delta_accuracy": 0.04347826086956519,
   "cw": 3,
   "wc": 9,
   "mean_token_reduction": 0.9025943852209082,
   "mean_evidence_count_reduction": 0.8709441172696811,
   "mean_reader_evidence_loss_rate": 0.9214795578829552,
   "confusion_transitions": {
    "RUMOR->NON_RUMOR": 9,
    "NON_RUMOR->RUMOR": 3
   }
  }
 }
}
```

## 10. Cutoff Analysis

```json
{
 "pheme": {
  "5": {
   "n": 50,
   "cw": 1,
   "wc": 2,
   "net_correction": 1,
   "mean_token_reduction": 0.48992900541517664,
   "mean_selected_count_reduction": 0.48034474206349215,
   "mean_static_units": 2.7,
   "mean_ms_units": 0.64,
   "mean_reader_evidence_loss_rate": 0.49317956349206354,
   "mean_proxy_margin_retention": 1.1107025359924512,
   "mean_confidence_delta": -0.003999999999999992,
   "mean_ms_cited_count": 0.64,
   "delta_macro_f1": 0.018677772655740776
  },
  "15": {
   "n": 50,
   "cw": 0,
   "wc": 5,
   "net_correction": 5,
   "mean_token_reduction": 0.593087935523161,
   "mean_selected_count_reduction": 0.5782891815786553,
   "mean_static_units": 4.16,
   "mean_ms_units": 1.06,
   "mean_reader_evidence_loss_rate": 0.5839266727424622,
   "mean_proxy_margin_retention": 1.195231733411958,
   "mean_confidence_delta": -0.041999999999999996,
   "mean_ms_cited_count": 1.06,
   "delta_macro_f1": 0.09939979505196894
  },
  "30": {
   "n": 50,
   "cw": 6,
   "wc": 5,
   "net_correction": -1,
   "mean_token_reduction": 0.7320807479660513,
   "mean_selected_count_reduction": 0.7083801439478594,
   "mean_static_units": 6.64,
   "mean_ms_units": 1.16,
   "mean_reader_evidence_loss_rate": 0.7283250986739358,
   "mean_proxy_margin_retention": 1.2546572849437962,
   "mean_confidence_delta": -0.049,
   "mean_ms_cited_count": 1.0,
   "delta_macro_f1": -0.026284405109374576
  },
  "60": {
   "n": 50,
   "cw": 3,
   "wc": 5,
   "net_correction": 2,
   "mean_token_reduction": 0.7206182140582866,
   "mean_selected_count_reduction": 0.7036916273714403,
   "mean_static_units": 6.64,
   "mean_ms_units": 1.4,
   "mean_reader_evidence_loss_rate": 0.6897966616716616,
   "mean_proxy_margin_retention": 1.1239244761232738,
   "mean_confidence_delta": -0.075,
   "mean_ms_cited_count": 1.24,
   "delta_macro_f1": 0.04001600640256098
  },
  "180": {
   "n": 50,
   "cw": 4,
   "wc": 2,
   "net_correction": -2,
   "mean_token_reduction": 0.7514552799499346,
   "mean_selected_count_reduction": 0.7350907848190458,
   "mean_static_units": 7.78,
   "mean_ms_units": 1.36,
   "mean_reader_evidence_loss_rate": 0.7261122633405243,
   "mean_proxy_margin_retention": 1.2376150455648132,
   "mean_confidence_delta": -0.09899999999999999,
   "mean_ms_cited_count": 1.36,
   "delta_macro_f1": -0.03832218506131557
  },
  "360": {
   "n": 50,
   "cw": 3,
   "wc": 4,
   "net_correction": 1,
   "mean_token_reduction": 0.7582874155611922,
   "mean_selected_count_reduction": 0.7353612374020536,
   "mean_static_units": 8.38,
   "mean_ms_units": 1.48,
   "mean_reader_evidence_loss_rate": 0.7460109618272882,
   "mean_proxy_margin_retention": 1.3766468715184166,
   "mean_confidence_delta": -0.08299999999999999,
   "mean_ms_cited_count": 1.48,
   "delta_macro_f1": 0.022186147186147087
  }
 },
 "maweibo": {
  "5": {
   "n": 50,
   "cw": 1,
   "wc": 1,
   "net_correction": 0,
   "mean_token_reduction": 0.7276299730938441,
   "mean_selected_count_reduction": 0.7130450158560206,
   "mean_static_units": 6.68,
   "mean_ms_units": 1.12,
   "mean_reader_evidence_loss_rate": 0.7195089632589632,
   "mean_proxy_margin_retention": 1.3051396430713817,
   "mean_confidence_delta": -0.0029999999999999914,
   "mean_ms_cited_count": 1.04,
   "delta_macro_f1": 0.001847290640394128
  },
  "15": {
   "n": 50,
   "cw": 3,
   "wc": 3,
   "net_correction": 0,
   "mean_token_reduction": 0.8574843014259058,
   "mean_selected_count_reduction": 0.8390053851490522,
   "mean_static_units": 10.46,
   "mean_ms_units": 0.86,
   "mean_reader_evidence_loss_rate": 0.8728496503496501,
   "mean_proxy_margin_retention": 1.2757512733198075,
   "mean_confidence_delta": -0.055,
   "mean_ms_cited_count": 0.86,
   "delta_macro_f1": -0.0013581190051779046
  },
  "30": {
   "n": 50,
   "cw": 3,
   "wc": 2,
   "net_correction": -1,
   "mean_token_reduction": 0.8618289608407909,
   "mean_selected_count_reduction": 0.833231197241775,
   "mean_static_units": 10.74,
   "mean_ms_units": 1.2,
   "mean_reader_evidence_loss_rate": 0.8380237586487587,
   "mean_proxy_margin_retention": 1.2407928127072296,
   "mean_confidence_delta": -0.05399999999999999,
   "mean_ms_cited_count": 1.1,
   "delta_macro_f1": -0.01904143299831995
  },
  "60": {
   "n": 50,
   "cw": 2,
   "wc": 1,
   "net_correction": -1,
   "mean_token_reduction": 0.8815137768878889,
   "mean_selected_count_reduction": 0.8425662789735857,
   "mean_static_units": 9.64,
   "mean_ms_units": 0.98,
   "mean_reader_evidence_loss_rate": 0.8877426175265309,
   "mean_proxy_margin_retention": 1.3771719301848686,
   "mean_confidence_delta": -0.066,
   "mean_ms_cited_count": 0.98,
   "delta_macro_f1": -0.01787278357932709
  },
  "180": {
   "n": 50,
   "cw": 4,
   "wc": 1,
   "net_correction": -3,
   "mean_token_reduction": 0.8896212431707008,
   "mean_selected_count_reduction": 0.8535179196157338,
   "mean_static_units": 11.6,
   "mean_ms_units": 1.3,
   "mean_reader_evidence_loss_rate": 0.8779216361869424,
   "mean_proxy_margin_retention": 1.1423734994859711,
   "mean_confidence_delta": -0.10800000000000001,
   "mean_ms_cited_count": 1.2,
   "delta_macro_f1": -0.060120201926924466
  },
  "360": {
   "n": 50,
   "cw": 0,
   "wc": 2,
   "net_correction": 2,
   "mean_token_reduction": 0.8881420327933828,
   "mean_selected_count_reduction": 0.844107279776695,
   "mean_static_units": 9.88,
   "mean_ms_units": 1.16,
   "mean_reader_evidence_loss_rate": 0.8802803192975607,
   "mean_proxy_margin_retention": 1.3251817805007955,
   "mean_confidence_delta": -0.053999999999999986,
   "mean_ms_cited_count": 1.16,
   "delta_macro_f1": 0.04099749020189647
  }
 }
}
```

## 11. Selection Pressure

```json
{
 "pheme": {
  "NO_CANDIDATE": {
   "n": 48,
   "cw": 0,
   "wc": 0,
   "net_correction": 0,
   "mean_token_reduction": null,
   "mean_selected_count_reduction": null,
   "mean_reader_evidence_loss_rate": null,
   "mean_proxy_margin_retention": 1.0,
   "mean_confidence_delta": 0.0
  },
  "LOW": {
   "n": 207,
   "cw": 10,
   "wc": 18,
   "net_correction": 8,
   "mean_token_reduction": 0.6411960140728645,
   "mean_selected_count_reduction": 0.6234847392669216,
   "mean_reader_evidence_loss_rate": 0.6238665588303269,
   "mean_proxy_margin_retention": 1.2376793198862979,
   "mean_confidence_delta": -0.05990338164251209
  },
  "MEDIUM": {
   "n": 40,
   "cw": 6,
   "wc": 5,
   "net_correction": -1,
   "mean_token_reduction": 0.8921205852030116,
   "mean_selected_count_reduction": 0.8748398721376661,
   "mean_reader_evidence_loss_rate": 0.8925916791541791,
   "mean_proxy_margin_retention": 1.3741673774294438,
   "mean_confidence_delta": -0.10874999999999999
  },
  "HIGH": {
   "n": 5,
   "cw": 1,
   "wc": 0,
   "net_correction": -1,
   "mean_token_reduction": 0.9424742446525126,
   "mean_selected_count_reduction": 0.9121478521478522,
   "mean_reader_evidence_loss_rate": 0.9503296703296703,
   "mean_proxy_margin_retention": 1.1545166128187998,
   "mean_confidence_delta": -0.17
  }
 },
 "maweibo": {
  "NO_CANDIDATE": {
   "n": 14,
   "cw": 0,
   "wc": 0,
   "net_correction": 0,
   "mean_token_reduction": null,
   "mean_selected_count_reduction": null,
   "mean_reader_evidence_loss_rate": null,
   "mean_proxy_margin_retention": 1.0,
   "mean_confidence_delta": 0.0
  },
  "LOW": {
   "n": 65,
   "cw": 4,
   "wc": 0,
   "net_correction": -4,
   "mean_token_reduction": 0.6714415007936143,
   "mean_selected_count_reduction": 0.6547193494994211,
   "mean_reader_evidence_loss_rate": 0.6509857236780312,
   "mean_proxy_margin_retention": 1.287392654370837,
   "mean_confidence_delta": -0.009230769230769225
  },
  "MEDIUM": {
   "n": 35,
   "cw": 0,
   "wc": 1,
   "net_correction": 1,
   "mean_token_reduction": 0.8721818699122,
   "mean_selected_count_reduction": 0.8554877429908487,
   "mean_reader_evidence_loss_rate": 0.8547846756418185,
   "mean_proxy_margin_retention": 1.2974553386936334,
   "mean_confidence_delta": -0.1
  },
  "HIGH": {
   "n": 186,
   "cw": 9,
   "wc": 9,
   "net_correction": 0,
   "mean_token_reduction": 0.9131744162693919,
   "mean_selected_count_reduction": 0.875150467095139,
   "mean_reader_evidence_loss_rate": 0.9157776743275335,
   "mean_proxy_margin_retention": 1.2915542342748436,
   "mean_confidence_delta": -0.06935483870967744
  }
 }
}
```

## 12. Static Utility vs Qwen Evidence Use

```json
{
 "pheme": {
  "n_evidence": 1815,
  "n_cited": 1398,
  "n_uncited": 417,
  "mean_utility_cited": 1.08984709938639,
  "mean_utility_uncited": 2.1993021714172776,
  "auc_utility_predicts_citation": 0.43701691007708854,
  "topk": {
   "top1_recall": 0.30374173886078704,
   "top1_n": 252,
   "top3_recall": 0.555910447577114,
   "top3_n": 252,
   "top5_recall": 0.7115891714106002,
   "top5_n": 252
  }
 },
 "maweibo": {
  "n_evidence": 2950,
  "n_cited": 2296,
  "n_uncited": 654,
  "mean_utility_cited": 4.682673687821105,
  "mean_utility_uncited": 0.09400562517663937,
  "auc_utility_predicts_citation": 0.6158296838538503,
  "topk": {
   "top1_recall": 0.1974771059834403,
   "top1_n": 286,
   "top3_recall": 0.43258308537481344,
   "top3_n": 286,
   "top5_recall": 0.6199784420555654,
   "top5_n": 286
  }
 }
}
```

## 13. MS Removal vs Qwen Citation

```json
{
 "pheme": {
  "all": {
   "n": 300,
   "p_remove_given_cited": 0.7932761087267525,
   "p_remove_given_uncited": 0.8609112709832134,
   "citation_removal_gap": -0.06763516225646093,
   "cited_total": 1398,
   "uncited_total": 417,
   "cited_removed": 1109,
   "uncited_removed": 359
  },
  "CW": {
   "n": 17,
   "p_remove_given_cited": 0.9159663865546218,
   "p_remove_given_uncited": 0.8717948717948718,
   "citation_removal_gap": 0.04417151475975001,
   "cited_total": 119,
   "uncited_total": 39,
   "cited_removed": 109,
   "uncited_removed": 34
  },
  "WC": {
   "n": 23,
   "p_remove_given_cited": 0.8682170542635659,
   "p_remove_given_uncited": 0.8775510204081632,
   "citation_removal_gap": -0.009333966144597361,
   "cited_total": 129,
   "uncited_total": 49,
   "cited_removed": 112,
   "uncited_removed": 43
  }
 },
 "maweibo": {
  "all": {
   "n": 300,
   "p_remove_given_cited": 0.9133275261324042,
   "p_remove_given_uncited": 0.9128440366972477,
   "citation_removal_gap": 0.00048348943515652554,
   "cited_total": 2296,
   "uncited_total": 654,
   "cited_removed": 2097,
   "uncited_removed": 597
  },
  "CW": {
   "n": 13,
   "p_remove_given_cited": 0.9313725490196079,
   "p_remove_given_uncited": 0.9411764705882353,
   "citation_removal_gap": -0.009803921568627416,
   "cited_total": 102,
   "uncited_total": 17,
   "cited_removed": 95,
   "uncited_removed": 16
  },
  "WC": {
   "n": 10,
   "p_remove_given_cited": 0.9746835443037974,
   "p_remove_given_uncited": 0.9411764705882353,
   "citation_removal_gap": 0.03350707371556216,
   "cited_total": 79,
   "uncited_total": 34,
   "cited_removed": 77,
   "uncited_removed": 32
  }
 }
}
```

## 14. Structural / Textual Characteristics

```json
{
 "structural": {
  "pheme": {
   "qwen_cited": {
    "n": 1398,
    "mean_utility": 1.08984709938639,
    "mean_relevance": 0.4723474509313252,
    "mean_depth": 1.3283261802575108,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1431.261087267525,
    "mean_reply_tokens": 25.09799713876967,
    "mean_parent_tokens": 31.96137339055794,
    "mean_pair_tokens": 78.05436337625179,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8283261802575107,
    "is_memory_previous_rate": 0.15808297567954221,
    "depth_group_rates": {
     "depth1": 0.8433476394849786,
     "depth2": 0.07582260371959942,
     "depth>=3": 0.08082975679542204
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.3726752503576538,
     "middle_50": 0.47067238912732473,
     "newest_quartile": 0.15665236051502146
    }
   },
   "ms_retained": {
    "n": 347,
    "mean_utility": 1.717125172949576,
    "mean_relevance": 0.4056575500154887,
    "mean_depth": 1.2853025936599423,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1223.7694524495678,
    "mean_reply_tokens": 22.04899135446686,
    "mean_parent_tokens": 29.337175792507203,
    "mean_pair_tokens": 72.45533141210375,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7089337175792507,
    "is_memory_previous_rate": 0.6138328530259366,
    "depth_group_rates": {
     "depth1": 0.7752161383285303,
     "depth2": 0.11527377521613832,
     "depth>=3": 0.10951008645533142
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.4553314121037464,
     "middle_50": 0.41786743515850144,
     "newest_quartile": 0.12680115273775217
    }
   },
   "ms_removed": {
    "n": 1468,
    "mean_utility": 1.256725351096509,
    "mean_relevance": 0.45790381470699765,
    "mean_depth": 1.4775204359673024,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1526.783378746594,
    "mean_reply_tokens": 25.862397820163487,
    "mean_parent_tokens": 31.384196185286104,
    "mean_pair_tokens": 78.2874659400545,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.7683923705722071,
    "is_memory_previous_rate": 0.03678474114441417,
    "depth_group_rates": {
     "depth1": 0.7915531335149864,
     "depth2": 0.09128065395095368,
     "depth>=3": 0.11716621253405994
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.2997275204359673,
     "middle_50": 0.5081743869209809,
     "newest_quartile": 0.19209809264305178
    }
   }
  },
  "maweibo": {
   "qwen_cited": {
    "n": 2296,
    "mean_utility": 4.682673687821105,
    "mean_relevance": 0.4066587065406212,
    "mean_depth": 1.1790069686411149,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1941.3837108013938,
    "mean_reply_tokens": 12.765243902439025,
    "mean_parent_tokens": 54.15766550522648,
    "mean_pair_tokens": 87.84756097560975,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.8793554006968641,
    "is_memory_previous_rate": 0.048344947735191636,
    "depth_group_rates": {
     "depth1": 0.8793554006968641,
     "depth2": 0.08449477351916376,
     "depth>=3": 0.03614982578397213
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.281794425087108,
     "middle_50": 0.48301393728222997,
     "newest_quartile": 0.23519163763066203
    }
   },
   "ms_retained": {
    "n": 256,
    "mean_utility": 0.48209840612253174,
    "mean_relevance": 0.36764087905046383,
    "mean_depth": 1.3125,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 1830.6640625,
    "mean_reply_tokens": 10.53125,
    "mean_parent_tokens": 53.47265625,
    "mean_pair_tokens": 84.86328125,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.69921875,
    "is_memory_previous_rate": 0.3671875,
    "depth_group_rates": {
     "depth1": 0.69921875,
     "depth2": 0.29296875,
     "depth>=3": 0.0078125
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.31640625,
     "middle_50": 0.4765625,
     "newest_quartile": 0.20703125
    }
   },
   "ms_removed": {
    "n": 2694,
    "mean_utility": 3.967884660035416,
    "mean_relevance": 0.4025869904934531,
    "mean_depth": 1.212694877505568,
    "mean_degree": 0.0,
    "mean_elapsed_seconds": 2085.4736451373424,
    "mean_reply_tokens": 12.724944320712694,
    "mean_parent_tokens": 51.52746844840386,
    "mean_pair_tokens": 85.21417965850037,
    "is_leaf_rate": 1.0,
    "is_source_child_rate": 0.852264291017075,
    "is_memory_previous_rate": 0.016332590942835932,
    "depth_group_rates": {
     "depth1": 0.852264291017075,
     "depth2": 0.10579064587973273,
     "depth>=3": 0.04194506310319228
    },
    "temporal_group_rates": {
     "oldest_quartile": 0.26280623608017817,
     "middle_50": 0.5025983667409057,
     "newest_quartile": 0.23459539717891612
    }
   }
  }
 },
 "textual": {
  "pheme": {
   "qwen_cited": {
    "n": 1398,
    "mean_reply_tokens": 25.09799713876967,
    "mean_parent_tokens": 31.96137339055794,
    "mean_pair_tokens": 78.05436337625179,
    "question_rate": 0.15021459227467812,
    "exclamation_rate": 0.11373390557939914,
    "url_rate": 0.17310443490701002,
    "mean_mention_count": 1.4492131616595136
   },
   "ms_removed": {
    "n": 1468,
    "mean_reply_tokens": 25.862397820163487,
    "mean_parent_tokens": 31.384196185286104,
    "mean_pair_tokens": 78.2874659400545,
    "question_rate": 0.15190735694822888,
    "exclamation_rate": 0.10694822888283378,
    "url_rate": 0.16076294277929154,
    "mean_mention_count": 1.5163487738419619
   },
   "cw_removed": {
    "n": 143,
    "mean_reply_tokens": 27.706293706293707,
    "mean_parent_tokens": 33.37762237762238,
    "mean_pair_tokens": 82.32167832167832,
    "question_rate": 0.07692307692307693,
    "exclamation_rate": 0.1048951048951049,
    "url_rate": 0.18181818181818182,
    "mean_mention_count": 1.6853146853146854
   },
   "wc_removed": {
    "n": 155,
    "mean_reply_tokens": 27.27741935483871,
    "mean_parent_tokens": 30.26451612903226,
    "mean_pair_tokens": 78.3741935483871,
    "question_rate": 0.18064516129032257,
    "exclamation_rate": 0.096774193548
```

## 15. PHEME vs Ma-Weibo

```json
{
 "pheme": {
  "n": 300,
  "static_macro_f1": 0.6278123637826781,
  "ms_macro_f1": 0.6475524475524477,
  "delta_macro_f1": 0.019740083769769545,
  "mean_static_social_tokens": 445.74666666666667,
  "mean_ms_social_tokens": 80.28666666666666,
  "mean_token_reduction": 0.6870030537478812,
  "mean_static_units": 6.05,
  "mean_ms_units": 1.1833333333333333,
  "mean_static_cited_count": 4.66,
  "mean_ms_cited_count": 1.13,
  "mean_reader_evidence_loss_rate": 0.6729987825225922,
  "mean_proxy_margin_retention": 1.2164629912591185,
  "mean_confidence_delta": -0.05866666666666666,
  "citation_removal_gap": -0.06763516225646093,
  "auc_utility_citation": 0.43701691007708854,
  "cw": 17,
  "wc": 23,
  "cutoff_delta_macro_f1": {
   "5": 0.018677772655740776,
   "15": 0.09939979505196894,
   "30": -0.026284405109374576,
   "60": 0.04001600640256098,
   "180": -0.03832218506131557,
   "360": 0.022186147186147087
  },
  "label_delta_accuracy": {
   "RUMOR": 0.017857142857142794,
   "NON_RUMOR": 0.021276595744680882
  },
  "cw_vs_cc_wc": {
   "cw_mean_reduction": 0.8834619597677035,
   "cc_wc_mean_reduction": 0.6985378101699578,
   "cw_mean_ms_units": 1.0,
   "cc_wc_mean_ms_units": 1.0666666666666667
  },
  "cw_loss_rate": 0.855162321338792,
  "cc_loss_rate": 0.6752640173835821,
  "wc_loss_rate": 0.8399526198439243,
  "proxy_margin_spearman_confidence": -0.08569219292038606,
  "proxy_margin_spearman_transition": 0.0750769778303865
 },
 "maweibo": {
  "n": 300,
  "static_macro_f1": 0.8825174825174824,
  "ms_macro_f1": 0.8731924360400445,
  "delta_macro_f1": -0.009325046477437926,
  "mean_static_social_tokens": 831.6766666666666,
  "mean_ms_social_tokens": 82.99333333333334,
  "mean_token_reduction": 0.8532185469392266,
  "mean_static_units": 9.833333333333334,
  "mean_ms_units": 1.1033333333333333,
  "mean_static_cited_count": 7.653333333333333,
  "mean_ms_cited_count": 1.0566666666666666,
  "mean_reader_evidence_loss_rate": 0.848133507382717,
  "mean_proxy_margin_retention": 1.2777351565450081,
  "mean_confidence_delta": -0.05666666666666668,
  "citation_removal_gap": 0.00048348943515652554,
  "auc_utility_citation": 0.6158296838538503,
  "cw": 13,
  "wc": 10,
  "cutoff_delta_macro_f1": {
   "5": 0.001847290640394128,
   "15": -0.0013581190051779046,
   "30": -0.01904143299831995,
   "60": -0.01787278357932709,
   "180": -0.060120201926924466,
   "360": 0.04099749020189647
  },
  "label_delta_accuracy": {
   "RUMOR": -0.05555555555555558,
   "NON_RUMOR": 0.04347826086956519
  },
  "cw_vs_cc_wc": {
   "cw_mean_reduction": 0.8909096300538507,
   "cc_wc_mean_reduction": 0.8512740324303241,
   "cw_mean_ms_units": 1.0,
   "cc_wc_mean_ms_units": 1.099236641221374
  },
  "cw_loss_rate": 0.9100050276520864,
  "cc_loss_rate": 0.8380884147727096,
  "wc_loss_rate": 0.9838888888888888,
  "proxy_margin_spearman_confidence": 0.02057297452754712,
  "proxy_margin_spearman_transition": -0.0928829726654661
 }
}
```

## 16. Cross-Fold Development Audit

```json
{
 "pheme": {
  "n_sampled_events": 285,
  "n_events_in_any_other_fold_test": 285,
  "rate_events_in_any_other_fold_test": 1.0,
  "n_events_in_multiple_validation_folds": 47,
  "n_samples": 300
 },
 "maweibo": {
  "n_sampled_events": 275,
  "n_events_in_any_other_fold_test": 275,
  "rate_events_in_any_other_fold_test": 1.0,
  "n_events_in_multiple_validation_folds": 53,
  "n_samples": 300
 }
}
```

## Root Causes

### OVER_COMPRESSION: not flagged

```json
{
 "flag": false,
 "cw_mean_reduction": 0.8909096300538507,
 "cc_wc_mean_reduction": 0.8512740324303242,
 "cw_mean_ms_units": 1.0,
 "cc_wc_mean_ms_units": 1.099236641221374,
 "rule": {
  "compression_gap": 0.05,
  "evidence_gap": 0.5
 }
}
```

### PROXY_READER_MARGIN_MISMATCH: FLAGGED

```json
{
 "flag": true,
 "mean_margin_retention": 1.2986399532742023,
 "delta_macro_f1": -0.009325046477437926,
 "rule": {
  "retention_min": 0.95,
  "delta_max": -0.005
 }
}
```

### READER_SENSITIVE_EVIDENCE_REMOVAL: not flagged

```json
{
 "flag": false,
 "cw_loss_rate": 0.9100050276520864,
 "cc_loss_rate": 0.8380884147727096,
 "wc_loss_rate": 0.9838888888888888,
 "rule": {
  "loss_rate_gap": 0.15,
  "loss_rate_ratio": 2.0
 }
}
```

### STATIC_UTILITY_READER_MISALIGNMENT: not flagged

```json
{
 "flag": false,
 "auc": 0.6158296838538503,
 "rule": {
  "auc_max": 0.6
 }
}
```

### DATASET_SPECIFIC_CONTEXT_NEED: FLAGGED

```json
{
 "flag": true,
 "detail": {
  "pheme": {
   "mean_token_reduction": 0.6870030537478812,
   "mean_reader_evidence_loss_rate": 0.6729987825225922,
   "delta_macro_f1": 0.019740083769769545
  },
  "maweibo": {
   "mean_token_reduction": 0.8532185469392266,
   "mean_reader_evidence_loss_rate": 0.848133507382717,
   "delta_macro_f1": -0.009325046477437926
  }
 },
 "note": "same MS-TSR transfers on PHEME but not on Ma-Weibo"
}
```

## Candidate Risk Signals

Diagnostic signals only; nothing here is promoted to a rule. Any future use requires fold-local prospective validation (§22).

- pheme: reader_evidence_loss_rate CW 0.8552 vs CC 0.6753 / WC 0.8400 (CW-CC 0.1799)
- maweibo: reader_evidence_loss_rate CW 0.9100 vs CC 0.8381 / WC 0.9839 (CW-CC 0.0719)
- proxy margin retention vs correctness transition: Spearman -0.0929 (Ma-Weibo), 0.0751 (PHEME) — no usable monotone relation.
- Static Utility -> Qwen citation AUC: 0.6158 (Ma-Weibo), 0.4370 (PHEME).
- MS removal vs citation gap: 0.0005 (Ma-Weibo), -0.0676 (PHEME).
- label asymmetry (Ma-Weibo delta accuracy): RUMOR -0.0556, NON_RUMOR 0.0435.
- fixed-threshold verdict: maweibo_signal=False, pheme_consistent=True.

## Protocol Implication

The 600-sample pilot merges five outer-fold validation pools: Ma-Weibo 275/275 and PHEME 285/285 sampled events also appear in another fold's test split (rate 1.0000), so the aggregate pilot cannot be used to design a global heuristic and still claim the original 5-fold test was untouched.

No stable proxy-side reader-risk signal cleared the fixed thresholds, so no fold-local reader calibration rule is justified at this point. Should one ever be introduced, it must be calibrated inside the outer fold only, leaving that fold's test untouched (§24).

## Recommendation

**MS_TSR_COMPRESSION_ONLY**

```json
{
 "cw_loss": 0.9100050276520864,
 "cc_loss": 0.8380884147727096,
 "wc_loss": 0.9838888888888888,
 "best_other_loss": 0.9838888888888888,
 "pheme_cw_loss": 0.855162321338792,
 "pheme_wc_loss": 0.8399526198439243,
 "maweibo_signal": false,
 "pheme_consistent": true,
 "reason": "compression is large and stable but no proxy-side signal predicts reader degradation"
}
```

## Verifier
issues = 0 (samples=600, groups={'pheme': {'CC': 172, 'CW': 17, 'WC': 23, 'WW': 88}, 'maweibo': {'CC': 252, 'CW': 13, 'WC': 10, 'WW': 25}}, root_causes=['PROXY_READER_MARGIN_MISMATCH', 'DATASET_SPECIFIC_CONTEXT_NEED'])
