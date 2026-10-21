# Surf spot finder

Train a model to spot clean, lined-up, directional breaking waves in aerial or
satellite images of Cornwall, then scan big-swell days to rank stretches of
coast that might hold a wave.

## 1. Where the imagery comes from

**Google Earth: look, don't train.** Google's terms ban bulk downloading or
scraping its imagery and using it to train machine-learning models. That
includes the current wavy-day Cornwall mosaic. Use Google Earth to browse,
check historic dates with the timeline slider in Google Earth Pro, and open the
`candidates.kml` this pipeline produces. Don't build the training set from it.

Sources you can actually train on:

| Source | Resolution | How often | Good for | Cost |
|---|---|---|---|---|
| **Sentinel-2** (Copernicus) | 10 m | every 2–5 days, about 11:00 UTC | where swell breaks, whitewater bands, how organised the lines are | free; `find_swell_scenes.py` gets it for you |
| **Channel Coastal Observatory** aerial photos (coastalmonitoring.org) | 10–25 cm | surveys every few years | wave shape and peel, sandbar and reef structure | free, Open Government Licence |
| **PlanetScope** | 3 m | daily | the sweet spot: sharp enough for wave shape and almost always a big-swell day available | Planet Education & Research programme, or paid |
| Your own drone or plane photos | any | whenever you like | ground truth for your local spots | yours |

At 10 m, Sentinel-2 can't resolve one peeling wave. It does show long-period
swell lines and where whitewater forms and fans out around points, reefs and
banks, which is what gives a spot away. Train one model per resolution. A model
trained on 25 cm aerial chips won't transfer to 10 m satellite chips.

## 2. Pipeline

```bash
pip install -r requirements.txt

# a) Find cloud-free Sentinel-2 passes on big-swell days and download crops
python find_swell_scenes.py --area north_coast --start 2023-10-01 --end 2024-03-31 \
    --min-swell 2.5 --max-cloud 25 --download scenes/
#   -> swell_scenes.csv lists swell size/period/direction, wind, an offshore flag, and cloud cover

# b) Cut images into chips (GeoTIFFs keep lat/lon; plain PNG/JPG work too)
python tile_images.py scenes/*.tif --out chips --size 64 --stride 32      # Sentinel-2, about 640 m chips
python tile_images.py aerial/*.tif --out chips_aerial --size 512 --stride 384   # 25 cm aerial, about 128 m chips

# c) Label in your browser: 1 clean, 2 messy, 3 no surf, s skip, b back
python label.py chips

# d) Fine-tune an ImageNet-pretrained ConvNeXt (use a GPU or Apple Silicon if you have one; CPU works, just slower)
python train.py --chips chips --epochs 15 --out model.pt

# e) Score a new big-swell day and rank candidates
python tile_images.py newday/*.tif --out newday_chips --size 64 --stride 32
python score.py --model model.pt --chips newday_chips --out results --heatmap
#   -> results/scores.csv     every chip ranked by P(clean), with lat/lon
#      results/top.html       gallery of the best chips
#      results/candidates.kml open in Google Earth to check each spot
#      results/*_heat.jpg     red overlay showing where it thinks clean waves are
```

## 3. Labelling tips

- **clean**: organised lines breaking in one direction, with a peel or shoulder
  and whitewater that starts at a point and runs along the wave instead of across the whole bay.
- **messy**: whitewater everywhere, close-outs, onshore chop, shapeless foam.
- **no surf**: flat water, land, beach, rocks, open ocean with no breaking.
- Aim for at least 150 to 200 **clean** examples spread over several days, tides
  and lighting. Clean chips are rare, so label every one you find. Training
  weights the classes to make up for the imbalance.
- Add known spots on good days (Fistral, Porthleven, Sennen and so on) as
  positives and known close-out beaches as messy. The model then learns the
  difference you care about.
- Training rotates and flips chips, so swell direction doesn't bias it.
  Conditions (swell direction, period, wind) come from `swell_scenes.csv`, not the pixels.

## 4. Turning scores into spots

A chip that scores high on one day might be a fluke. A spot that scores high
on many big-swell days under a particular swell direction and wind is a real
candidate. Join `scores.csv` from several days on lat/lon, then group by the
conditions in `swell_scenes.csv`. Check the best ones in Google Earth and on
UKHO/EA bathymetry or LiDAR to see if a reef or point explains the wave.

## Files

- `common.py` holds the class names, model builder and augmentations
- `tile_images.py` turns images into chips plus `index.csv` (with lat/lon for GeoTIFFs)
- `label.py` is the browser labeller and writes `labels.csv`
- `train.py` fine-tunes the model and prints the validation confusion matrix
- `score.py` writes ranked scores, a gallery, KML and heatmaps
- `find_swell_scenes.py` combines Open-Meteo swell and wind history with the Sentinel-2 catalogue

Tested end to end on synthetic wave images: tiling, the labeller, training,
scoring, GeoTIFF lat/lon and the KML output all work. `find_swell_scenes.py`
couldn't run here because the build machine has no access to Open-Meteo or
the satellite catalogue, so it hasn't been tested yet. Run it on your own computer.

## 5. Seabed depth from the colour of the sea (`seabed_depth.py`)

Blue light reaches deeper into water than green, so the blue/green balance of
the sea in a satellite image changes with depth. This is satellite-derived
bathymetry (the Stumpf ratio method plus a gradient-boosted model). The model
learns that relationship from places where the depth is already surveyed, then
maps the rest of the coast. It works to about 15 m in clear water, which covers
the reefs, points and sandbars that shape a wave.

```bash
pip install scikit-learn   # plus rasterio and requests from requirements.txt

# 1. Calm, clear, flat Sentinel-2 days (low swell, light wind, <10% cloud)
python seabed_depth.py fetch --area penwith --start 2023-04-01 --end 2023-09-30 --out calm/

# 2. Known depths: download a surveyed depth tile for the same area, then sample it
#    EMODnet Bathymetry (emodnet.ec.europa.eu, about 115 m grid, covers everything)
#    UKHO Admiralty Marine Data Portal (datahub.admiralty.co.uk, detailed multibeam where surveyed)
#    Environment Agency LiDAR (environment.data.gov.uk/survey, reefs and sand that dry at low tide)
python seabed_depth.py sample emodnet_tile.tif --elevation --out known_depths.csv

# 3. Train and map depth
python seabed_depth.py train --scenes "calm/*.tif" --depths known_depths.csv --out depth/
#   -> depth/depth.tif   band 1 depth in metres (NaN where the bottom isn't visible),
#                        band 2 how many clear scenes saw each pixel
#      depth/depth.png   quick look (light = shallow)
#      depth/report.txt  accuracy on held-out stretches of coast, classic Stumpf vs boosted
```

Things to know:
- **Use calm days for depth and big-swell days for waves.** Swell stirs up
  sand and foam, which hides the bottom. Foam, glint, cloud and water too deep
  to see the bottom are all masked out instead of guessed.
- **Tide.** Depth is measured from the water level when the image was taken,
  and Cornwall's tide moves by up to about 6 m. For now, pick scenes near the
  same tide, or add the tide height to the result. Correcting for tide
  automatically is the next step.
- **Expected accuracy:** about 1 to 2 m in Cornwall's real water. On the
  synthetic test it was 0.3 m, but that test water was perfectly clear.
- **Getting the depth deeper:** below about 15 m the colour stops changing.
  Big-swell images can still give depth there, because waves bunch up and slow
  down over shallower water. That method is a later step.

## 6. Point-break finder (`point_breaks.py`)

This combines the seabed map with wave physics and a small learned model, then
ranks every 100 m of coast by how likely it is to hold a peeling point or reef
wave on a big swell.

```bash
pip install scipy scikit-learn

# 1. Merge colour-derived shallow depth with surveyed deep water (EMODnet)
python point_breaks.py prepare --sdb depth/depth.tif --fill emodnet_tile.tif --out seabed.tif

# 2. Trace big swell from several directions and rank the coast
python point_breaks.py scan --seabed seabed.tif --known known_breaks.csv \
    --swell-dirs 250 270 290 310 --period 14 --height 3 --out breaks/
#   -> breaks/candidates.csv  ranked spots: best swell direction, peel angle, ride length
#      breaks/candidates.kml  open in Google Earth
#      breaks/map.png         seabed, swell paths, break points (green = surfable peel,
#                             red = closeout, yellow = too steep), top 15 circled
#      breaks/report.txt      the hidden-break test
```

How it works:
- **Physics.** Swell paths bend towards shallow water as they cross the
  seabed, grow, and break at about 0.78 of the depth. At each break it works
  out the *peel angle*, the angle between the wave crest and the line it breaks
  along. Under 25° is a closeout. 30° to 70° is a surfable peel. A long run of
  surfable peel is a point break.
- **Learning.** Every coastal point gets features: how far the land sticks
  out, seabed depth and slope at 100 m to 1.2 m offshore, how rough the
  seabed is, and the physics results for each swell direction. A deliberately
  simple model learns which of these patterns match the reefs and points in
  `known_breaks.csv`. Beaches there count as examples of what isn't a point break.
- **Honest test.** Each known reef or point is hidden in turn, and the report
  says where it ranks along the coast without the model having seen it.
- The final score is half physics and half learned. Spots far from any known
  break can still rank high on physics alone, and that's where unknown spots show up.

Things to know:
- `known_breaks.csv` lists 51 Cornish breaks with coordinates and types from
  public surf guides (mondo.surf, surfnerd): 20 reef and point breaks, 20
  beaches, and 11 marked `mixed` or `harbour`. Reefs and points are what the
  model learns to find. Beaches are what it learns to avoid. Mixed and harbour
  spots are left out of training. Open `known_breaks.kml` in Google Earth to
  check them. Changing a type or adding a spot you know improves the model the most.
- The seabed needs to reach deep water (40 m or more) on the side the swell
  comes from, so always merge in EMODnet.
- Use `--tide` to test different tide stages, for example `--tide 3` for
  roughly mid tide on a low-water chart datum. Many reefs only work at one
  stage of the tide.
- This is simple ray tracing with no wave diffraction, currents or wind, so
  treat the output as a shortlist to check in Google Earth and in person, not
  a guarantee.
- On a synthetic coast with four headlands, it ranked the three with a gentle
  reef shelf as the top three. The steep-sided one ranked sixth. Each hidden
  headland ranked in roughly the top 10% of the coast
  ([example map](example_synthetic_point_breaks.png)).
