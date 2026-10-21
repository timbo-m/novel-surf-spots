# Cornwall candidates vs proven breaks

There are 30 candidates (the top 15 from Trevose and the top 15 from Bude) and 14 proven reference breaks:
- Boilers and Anchor Point in Morocco (both rights), from mondo.surf
- Porthleven
- The proven reefs and points in known_breaks.csv for the two stretches: Cribbar, Constantine Reef, Treyarnon, Daymer Point, Black Rock, Duckpool, Millook, Tremoutha Haven, Foxhole, Marsland Left and Compass Point

The method is the same as in compare_pentire_boilers, minus the satellite-imagery component, which is too slow and too weak at 10 m for 44 spots. With imagery dropped, the weights are renormalised to: geometry 0.1875, seabed 0.25, physics 0.375, climate 0.1875.

The model traces a 3 m, 14 s swell from 230–340° every 10°; for Porthleven it uses 190–300°.

When picking each candidate's closest proven match, references within 1 km of the candidate are skipped, so a candidate on a known spot doesn't just match itself.

## Left or right
For each swell direction, I take the rays that break within 400 m of the take-off and fit break time against position along the coast. If the break moves towards a surfer's right as they ride in, the spot is a right. Each direction's vote is weighted by its physics score. A vote of 40% or less one way is called "peak (both ways)".

How it did on spots whose direction is known:
- Boilers and Anchor Point: both called right. Correct.
- Known lefts with a modelled wave: all six called left (Treyarnon, Millook, Tremoutha Haven, Foxhole, Marsland Left, Compass Point). Foxhole's call is weak.
- Two-way spots: Cribbar, Black Rock and Porthleven were called right, Duckpool "both ways". The method gives one dominant direction, so "right" here means the right is the stronger side.
- No call: Constantine Reef and Daymer Point. The model traces no surfable break at either.

## Validation of the score
- Boilers and Anchor Point are each other's closest match, at 66.
- Proven Cornish reefs score 6–52 against Boilers (median about 37). Treyarnon is the closest at 52.
- Daymer and Constantine score low against everything, because the model finds no wave at them. Treat them as weak references.

## Files
- ranked_candidates.csv: every candidate with its left/right call, best swell direction, closest proven match and score, its scores against Boilers and Anchor Point, and its scores against every reference
- references_matrix.csv: the references scored against each other, with known vs called direction
- candidates_matched.kml: open in Google Earth, or import into Google My Maps. Red stars are rights, blue stars lefts, yellow stars "peak (both ways)".
- compare_many.py re-runs everything; physics is cached in cache/

## Caveats
- These are model estimates: ray tracing with no diffraction, no tide and one swell size. Many directions rest on only a handful of breaking rays.
- Seabed quality varies by area, measured as held-out R² of the colour-depth model:
  - Trevose 0.76
  - Anchor Point 0.70
  - Bude 0.63
  - Porthleven 0.38
  - Boilers 0.36
- Similarity measures shape and setting, not quality. A high score means "built like X", not "as good as X".
