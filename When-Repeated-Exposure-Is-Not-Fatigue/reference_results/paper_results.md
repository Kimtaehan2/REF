# Results reported in the manuscript

All exposure coefficients are reported per one standard deviation. Skip and cessation effects are odds ratios; watch-intensity effects are linear mixed-model coefficients among non-skipped interactions.

## Representative behavioral associations

| Exposure measure | Skip OR (95% CI) | p | Watch beta (95% CI) | p | Cessation OR | Within-user OR | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| (a) Category streak | 1.037 (0.990, 1.085) | .122 | 0.062 (0.060, 0.063) | <.001 | 1.017 | 1.005 | Mixed |
| (b) Time-gated category, 20 s | 1.185 (1.147, 1.224) | <.001 | -0.021 (-0.023, -0.019) | <.001 | 1.040 | 1.051 | Fatigue-consistent |
| (c) Temporal burst, 20 s | 1.172 (1.139, 1.206) | <.001 | -0.059 (-0.060, -0.057) | <.001 | 0.994 | 0.976 | Item rejection only |
| (f-2) Rolling cumulative, 30 min | 1.062 (1.011, 1.116) | .017 | 0.052 (0.050, 0.053) | <.001 | 0.795 | 0.738 | Sustained engagement |
| (h) Time-decayed exposure, 30 min | 1.087 (1.042, 1.133) | <.001 | 0.010 (0.008, 0.012) | <.001 | 0.780 | 0.636 | Sustained engagement |

## Complete skip and watch-intensity results

| Exposure | Skip OR | p | Watch beta | p | AIC | BIC | Grouped AUC | GEE converged | Mixed model converged |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| (a) category-only [baseline] | 1.0366 | .122 | 0.06188 | <.001 | 1255678.7 | 1255752.9 | 0.76390 | True | True |
| (b) time-gated T=20s | 1.1851 | <.001 | -0.02103 | <.001 | 1249163.3 | 1249237.5 | 0.76894 | True | True |
| (c) temporal burst T=20s | 1.1721 | <.001 | -0.05863 | <.001 | 1248347.5 | 1248421.7 | 0.77141 | True | True |
| (d) Jaccard overlap theta=0.5 | 1.0375 | .115 | 0.06190 | <.001 | 1255680.9 | 1255755.1 | 0.76401 | True | True |
| (e) combined theta=0.5 & T=20s | 1.1849 | <.001 | -0.02393 | <.001 | 1249092.6 | 1249166.8 | 0.76915 | True | True |
| (f-1) session cumulative | 1.0081 | .700 | 0.06778 | <.001 | 1255628.0 | 1255702.2 | 0.76517 | False | True |
| (f-2) rolling cumulative W=30m | 1.0619 | .017 | 0.05168 | <.001 | 1255554.8 | 1255629.0 | 0.76498 | True | True |
| (f-2) rolling cumulative W=60m | 1.0305 | .203 | 0.06074 | <.001 | 1255717.6 | 1255791.8 | 0.76470 | False | True |
| (f-2) rolling cumulative W=180m | 1.0071 | .770 | 0.07219 | <.001 | 1255663.3 | 1255737.5 | 0.76505 | False | True |
| (g) sliding density N=20 | 1.0436 | .005 | 0.04064 | <.001 | 1255619.9 | 1255694.1 | 0.76553 | True | True |
| (g) sliding density N=50 | 1.0209 | .231 | 0.04791 | <.001 | 1255710.6 | 1255784.8 | 0.76513 | True | True |
| (g) sliding density N=100 | 1.0123 | .502 | 0.05182 | <.001 | 1255644.7 | 1255718.9 | 0.76497 | True | True |
| (h) time-decay gauge tau=1800s | 1.0866 | <.001 | 0.00999 | <.001 | 1254923.7 | 1254997.9 | 0.76545 | True | True |
| (h) time-decay gauge tau=7200s | 1.1242 | <.001 | 0.03813 | <.001 | 1255134.4 | 1255208.6 | 0.76516 | True | True |
| (h) time-decay gauge tau=86400s | 1.1283 | <.001 | 0.05650 | <.001 | 1255178.8 | 1255253.0 | 0.76542 | True | True |
| (i) event-decay T=20s penalty=0.2 | 1.0105 | .697 | 0.00766 | <.001 | 1255143.6 | 1255217.8 | 0.76403 | False | True |
| (i) event-decay T=20s penalty=0.5 | 1.0159 | .573 | 0.01113 | <.001 | 1255152.9 | 1255227.1 | 0.76378 | True | True |
| (i) event-decay T=20s penalty=1.0 | 1.0168 | .580 | 0.01690 | <.001 | 1255230.8 | 1255305.0 | 0.76351 | True | True |
| (i) event-decay T=60s penalty=0.2 | 1.0344 | .237 | 0.01860 | <.001 | 1254822.8 | 1254897.0 | 0.76476 | True | True |
| (i) event-decay T=60s penalty=0.5 | 1.0508 | .114 | 0.02935 | <.001 | 1254846.5 | 1254920.7 | 0.76462 | True | True |
| (i) event-decay T=60s penalty=1.0 | 1.0647 | .067 | 0.04250 | <.001 | 1254945.3 | 1255019.5 | 0.76409 | True | True |

## Complete cessation results

| Exposure | Cessation OR | p | Within-user OR | p | Grouped AUC | Cessation converged | Within model converged |
|---|---:|---:|---:|---:|---:|---|---|
| (a) category-only streak | 1.0165 | <.001 | 1.0049 | .199 | 0.76615 | True | True |
| (b) time-gated T=20s | 1.0404 | <.001 | 1.0509 | <.001 | 0.76828 | True | True |
| (c) burst T=20s | 0.9938 | .023 | 0.9758 | <.001 | 0.76730 | True | True |
| (d) Jaccard >= 0.5 | 1.0165 | <.001 | 1.0046 | .213 | 0.76613 | True | True |
| (e) Jaccard >= 0.5 & T=20s | 1.0402 | <.001 | 1.0503 | <.001 | 0.76826 | True | True |
| (f-1) inferred-session cumulative | 1.0270 | <.001 | 1.1101 | <.001 | 0.76620 | True | True |
| (f-2) rolling cumulative W=30m | 0.7948 | <.001 | 0.7384 | <.001 | 0.76888 | True | True |
| (f-2) rolling cumulative W=60m | 0.7858 | <.001 | 0.7143 | <.001 | 0.76856 | True | True |
| (f-2) rolling cumulative W=180m | 0.9734 | .235 | 0.6464 | <.001 | 0.76822 | True | True |
| (g) sliding density N=20 | 1.0070 | .006 | 1.0070 | .004 | 0.76751 | True | True |
| (g) sliding density N=50 | 1.0058 | .030 | 1.0058 | .021 | 0.76748 | True | True |
| (g) sliding density N=100 | 1.0050 | .063 | 1.0052 | .044 | 0.76748 | True | True |
| (h) time-decay gauge tau=1800s | 0.7801 | <.001 | 0.6364 | <.001 | 0.76998 | True | True |
| (h) time-decay gauge tau=7200s | 0.8513 | <.001 | 0.6607 | <.001 | 0.76959 | True | True |
| (h) time-decay gauge tau=86400s | 0.3220 | <.001 | 0.5166 | <.001 | 0.77439 | True | False |
| (i) event-decay T=20s penalty=0.2 | 1.0561 | <.001 | 1.0730 | <.001 | 0.76934 | True | True |
| (i) event-decay T=20s penalty=0.5 | 1.0486 | <.001 | 1.0594 | <.001 | 0.76898 | True | True |
| (i) event-decay T=20s penalty=1.0 | 1.0359 | <.001 | 1.0404 | <.001 | 0.76838 | True | True |
| (i) event-decay T=60s penalty=0.2 | 1.0527 | <.001 | 1.0733 | <.001 | 0.76907 | True | True |
| (i) event-decay T=60s penalty=0.5 | 1.0345 | <.001 | 1.0419 | <.001 | 0.76821 | True | True |
| (i) event-decay T=60s penalty=1.0 | 1.0301 | <.001 | 1.0383 | <.001 | 0.76744 | True | True |

## Session-boundary sensitivity

| Exposure | 15m | 30m | 60m | data-derived 35.50m | user-adaptive |
|---|---:|---:|---:|---:|---:|
| (a) | 1.0334 | 1.0196 | 1.0101 | 1.0165 | 0.9949 |
| (b) | 1.0490 | 1.0416 | 1.0381 | 1.0404 | 1.0459 |
| (c) | 0.9630 | 0.9903 | 0.9972 | 0.9938 | 0.9678 |
| (d) | 1.0332 | 1.0195 | 1.0101 | 1.0165 | 0.9951 |
| (e) | 1.0482 | 1.0413 | 1.0380 | 1.0402 | 1.0457 |
| (f-1) | 1.0451 | 1.0307 | 1.0186 | 1.0270 | 0.9974 |
| (f-2) | 0.8240 | 0.8061 | 0.7613 | 0.7948 | 0.6567 |
| (g) | 1.0066 | 1.0068 | 1.0065 | 1.0070 | 1.0051 |
| (h) | 0.6811 | 0.7766 | 0.7510 | 0.7801 | 0.5680 |

## Threshold estimates

| Exposure | Threshold | 95% CI | F | Wild-bootstrap p | Threshold after excluding 50 users |
|---|---:|---:|---:|---:|---:|
| (a) category-only | 6 | [4, 8] | 3768.4 | 0.0000 | 4 |
| (b) time-gated T=20s | 2 | [2, 9] | 2392.1 | 0.0000 | 9 |
| (b) time-gated T=60s | 2 | [1, 9] | 1097.2 | 0.0000 | 9 |
| (d) jaccard theta=0.34 | 6 | [6, 9] | 5367.5 | 0.0000 | 5 |
| (d) jaccard theta=0.5 | 6 | [6, 9] | 5362.0 | 0.0000 | 5 |
| (d) jaccard theta=0.67 | 4 | [4, 6] | 2724.2 | 0.0000 | 9 |
| (d) jaccard theta=1.0 | 4 | [4, 6] | 2723.6 | 0.0000 | 9 |
| (e) combo theta=0.5,T=20 | 2 | [2, 9] | 2222.6 | 0.0000 | 9 |
| (e) combo theta=0.34,T=60 | 1 | [1, 9] | 994.1 | 0.0000 | 9 |
