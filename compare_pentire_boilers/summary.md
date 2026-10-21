# Pentire Point (Polzeath) vs Boilers (Cap Ghir): similarity 47 / 100

Spots: Pentire Point at 50.5852, -4.9319 (surf-finder candidate). Boilers at 30.62456, -9.87758 (mondo.surf).
Take-off points used are the best-scoring coastline within 200 m of those coordinates: Pentire 184 m away, Boilers 141 m away.

| component | weight | score | what it compares |
|---|---|---|---|
| geometry | 0.15 | 49 | Both stick out about the same amount: land fraction within 800 m is 0.39 vs 0.44. But Pentire's headland axis is 43° off its dominant swell, while Boilers' axis is 115° off. Boilers sits in the lee of Cap Ghir and needs swell to wrap. |
| seabed | 0.20 | 26 | Similar depth 500 m out (9.1 vs 8.5 m). Boilers then drops to 46 m within 2 km, while off Pentire the bay stays 10–13 m deep. Pentire's seabed is also rougher (2.8 vs 1.8). |
| physics | 0.30 | 65 | Best swell direction is 320° for Pentire and 280° for Boilers. Working windows: Pentire works at 270/290/320°, Boilers at 240/270/280/310°. Peel at the best direction is 55° vs 34°. Aligned on each spot's best direction, the peel curves score 0.52 similar and the breaker-height curves 0.67. |
| imagery | 0.20 | 43 | Big-swell Sentinel-2 days: 3 for Pentire, 6 for Boilers, but only 2 each had readable crests. Crests wrap about 53° at Pentire vs 23° at Boilers. Whitewater within 600 m of the take-off: 15% vs 12%. |
| climate | 0.15 | 47 | Swell of at least 1.5 m and 9 s from the working window: 11% vs 7% of daytime hours (ERA5, 2022–25). The same with offshore or light wind: only 2% at Pentire vs 5% at Boilers. Dominant swell: 267° vs 319°. |

Overall = weighted mean of the component scores, 47.5/100. With equal weights it would be 46.

## How to read it
- At about 47/100 they're partly alike. The headlands are a similar size, the near-shore depth is similar and the breaker heights are similar. The real differences:
  - Boilers drops to deep water quickly; Pentire sits behind a wide shallow bay.
  - Boilers is a wrap-around spot in the lee of a cape; Pentire faces the swell more directly.
  - Boilers gets good wind with its swell more than twice as often.
- Pentire's best modelled swell is NW (320°). Its main swell comes from the W (267°), which isn't in its best window.

## Uncertainty (read before trusting the number)
- **Known wave vs model candidate.** Boilers is a proven wave. Pentire is only a candidate from my model.
- **Boilers seabed is weaker.** The colour-depth model there used 4 calm days and scored R² 0.36 (3.5 m RMSE on held-out coast). Pentire's used 9 days and scored R² 0.76 (2.2 m). EMODnet off Morocco is probably interpolated from coarse surveys.
- **The physics curves are noisy.** Each direction rests on only 2–12 breaking rays near the take-off, so curves are compared by median difference, not RMS. Using RMS gave 39/100 overall, so treat the score as roughly 40–50.
- **The imagery is weak evidence.** At 10 m, Sentinel-2 barely resolves swell crests near the shore. Only 2 days per spot gave a readable crest direction, so crest spacing is reported but not scored.
- **No tide, wind-sea or diffraction modelling.** It's a 3 m, 14 s swell only. Ray tracing without diffraction understates how swell wraps into the lee of Cap Ghir, which is how Boilers actually gets its waves.

## Files
- side_by_side.png: both seabeds with swell rays at each spot's best direction, plus the peel, physics-score, breaker-height and depth-profile curves
- big_swell_days.png: the big-swell Sentinel-2 chips used
- component_scores.csv, imagery_days.csv, result.json
- compare_spots.py re-runs the whole comparison; intermediate results are cached in cache/
