# Error analysis & calibration by segment (temporal test)

## Conformal coverage check
```
{
  "A": {
    "target_coverage": 0.9,
    "empirical_coverage": 0.925,
    "abstain_rate(setsize!=1)": 0.815
  },
  "B": {
    "target_coverage": 0.9,
    "empirical_coverage": 0.913,
    "abstain_rate(setsize!=1)": 0.076
  }
}
```

## Per-segment metrics
| model   | segment                  |     n |   base_rate |    auc |   brier |    ece |
|:--------|:-------------------------|------:|------------:|-------:|--------:|-------:|
| A       | ALL                      | 31184 |       0.499 | 0.5586 |  0.2472 | 0.0159 |
| A       | topic=Beauty_Fashion     |  8449 |       0.483 | 0.5333 |  0.2496 | 0.0209 |
| A       | topic=Others             |  5268 |       0.512 | 0.5625 |  0.2468 | 0.0203 |
| A       | topic=Lifestyle          |  5020 |       0.472 | 0.5734 |  0.2462 | 0.0367 |
| A       | topic=Life_hacks_Persona |  3370 |       0.518 | 0.5678 |  0.2457 | 0.0108 |
| A       | topic=Dance_Music        |  2630 |       0.478 | 0.5683 |  0.2456 | 0.0291 |
| A       | topic=Sports_Fitness     |  1864 |       0.527 | 0.5676 |  0.2455 | 0.0161 |
| A       | topic=Shopping_Products  |  1805 |       0.573 | 0.5308 |  0.2488 | 0.0655 |
| A       | topic=Cooking_Food       |  1638 |       0.51  | 0.5667 |  0.2461 | 0.0173 |
| A       | creator=small            |  7796 |       0.502 | 0.5714 |  0.2454 | 0.007  |
| A       | creator=mid              |  7798 |       0.519 | 0.5563 |  0.2475 | 0.0109 |
| A       | creator=large            |  7796 |       0.494 | 0.5447 |  0.2486 | 0.0207 |
| A       | creator=xlarge           |  7794 |       0.482 | 0.5699 |  0.2474 | 0.0448 |
| A       | dur=<7s                  |  2256 |       0.537 | 0.6145 |  0.2359 | 0.0171 |
| A       | dur=7-15s                |  8862 |       0.498 | 0.5541 |  0.2484 | 0.0277 |
| A       | dur=15-30s               |  7611 |       0.493 | 0.5451 |  0.2485 | 0.016  |
| A       | dur=30-60s               |  6017 |       0.489 | 0.5461 |  0.2485 | 0.0147 |
| A       | dur=>60s                 |  6438 |       0.504 | 0.5638 |  0.2469 | 0.0095 |
| B       | ALL                      | 31184 |       0.499 | 0.9521 |  0.0841 | 0.0274 |
| B       | topic=Beauty_Fashion     |  8449 |       0.483 | 0.9414 |  0.0944 | 0.0256 |
| B       | topic=Others             |  5268 |       0.512 | 0.9513 |  0.0859 | 0.0417 |
| B       | topic=Lifestyle          |  5020 |       0.472 | 0.9557 |  0.0808 | 0.0323 |
| B       | topic=Life_hacks_Persona |  3370 |       0.518 | 0.9545 |  0.0812 | 0.0325 |
| B       | topic=Dance_Music        |  2630 |       0.478 | 0.9477 |  0.0872 | 0.0228 |
| B       | topic=Sports_Fitness     |  1864 |       0.527 | 0.9698 |  0.0644 | 0.0347 |
| B       | topic=Shopping_Products  |  1805 |       0.573 | 0.9586 |  0.0774 | 0.0249 |
| B       | topic=Cooking_Food       |  1638 |       0.51  | 0.9587 |  0.0798 | 0.03   |
| B       | creator=small            |  7796 |       0.502 | 0.9529 |  0.0851 | 0.0307 |
| B       | creator=mid              |  7798 |       0.519 | 0.9538 |  0.0833 | 0.0255 |
| B       | creator=large            |  7796 |       0.494 | 0.9428 |  0.0926 | 0.0298 |
| B       | creator=xlarge           |  7794 |       0.482 | 0.9585 |  0.0755 | 0.029  |
| B       | dur=<7s                  |  2256 |       0.537 | 0.9597 |  0.0777 | 0.0304 |
| B       | dur=7-15s                |  8862 |       0.498 | 0.9502 |  0.0881 | 0.038  |
| B       | dur=15-30s               |  7611 |       0.493 | 0.952  |  0.0837 | 0.026  |
| B       | dur=30-60s               |  6017 |       0.489 | 0.9531 |  0.0807 | 0.0183 |
| B       | dur=>60s                 |  6438 |       0.504 | 0.9517 |  0.0846 | 0.0298 |

Reliability plots: `reports/plots/reliability_A.png`, `reliability_B.png`.