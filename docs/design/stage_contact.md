# Stage `contact`: which object is in which hand (2026-09-26)

`src/duet/playground/contact.py`. Per ego stream with `derived/hands/<stream>.npz` (MediaPipe, 21 landmarks) and
`derived/objects/<stream>.npz` (YOLO-World boxes): per frame, per hand, the held object; smoothed; grasp/release events.
Runs in < 1 s on the 466-frame clip.

## Method

Per frame, per hand (index 0 = MediaPipe "Left", 1 = "Right", as the `hands` stage stores them):

- **candidates**: objects whose box, enlarged by 10 % about its centre, contains >= 3 of the 5 fingertips (landmarks 4,
  8, 12, 16, 20) or the palm centre (mean of landmarks 0, 5, 9, 13, 17)
- **tie-break**: smallest box area
- **raw score**: fraction of fingertips inside x object confidence. Deviation: a palm-only candidate with no fingertip
  inside gets fraction 0.2 instead of 0, so the score is never 0 for a held object.
- **depth check** (only when `episode/zed/<stream>/depth/*.png` and `export.json` exist): median depth at the 21 landmarks
  vs median depth inside the object box (depth png in mm, mapped to the extracted frame by the same time formula as
  `world.py`); a candidate is rejected when they differ by more than 15 cm. `<stream>.npz: depth_used` and the report say
  whether it ran. The CoMind clip has no ZED data, so on the clip it did not run; it is exercised by a synthetic test.

Temporal, per hand: majority vote of the raw label over a centred 0.5 s window (5 frames at 10 fps; ties keep the
current frame's label); then hysteresis: a contact starts after 0.3 s (3 frames) of the same voted label and ends after
0.5 s (5 frames) of a different label. Held labels are back-filled to the first frame of the run that triggered the
transition, so `held[]` and the events agree; changing object = release + grasp at the same frame. The smoothed score is
the mean raw score of the held object over the vote window.

Outputs: `derived/contact/<stream>.npz` (`held [N,2]`, `score [N,2]`, `raw_held`, `raw_score`, `depth_used`),
`derived/contact/events.json` (`{"t": reference seconds, "frame", "stream", "person", "hand": "L"|"R", "object",
"type": "grasp"|"release"}`), `records_extra.parquet` with `<stream>_held_L/R`, `<stream>_contact_score_L/R` (the
exporter prefixes `contact_`), `report.json`.

## Validation on `comind_43276420_clip`

Command: `.venv/bin/python scripts/playground.py run comind_43276420_clip --stages contact --force`

| stream | person | hand present L / R | contact any hand (smoothed) | raw any hand | L / R | events | objects held (frames) |
|---|---|---|---|---|---|---|---|
| leader | leader | 58 % / 71 % | 39.9 % | 44.6 % | 14.6 % / 36.5 % | 39 | bowl 161, pan 23, strainer 18, pot 11, cutting board 11, lid 11, bread 3 |
| helper | helper | 60 % / 52 % | 43.1 % | 49.1 % | 25.8 % / 26.4 % | 48 | bowl 98, pan 69, cutting board 46, bread 21, sink 9 |

87 events in 46.6 s; full list in `derived/contact/events.json`. Many are short (0.3-0.5 s, the minimum the hysteresis
allows) and the same steel bowl is labelled bowl / pan / pot / strainer by YOLO-World, so an object "switch" often is
the same object under another name.

Qualitative check, 4 frames rendered with landmarks (fingertips as dots) and the chosen box
(`scratchpad/contact_examples.jpg`, `contact_<stream>_<k>.png`):

- `helper` k=175 (t=19.2 s): R hand entering from the bottom of the frame holds the steel bowl, labelled "pan" 0.35:
  correct contact, wrong class name. The partner's arm with a whisk in the bowl is not detected as a hand.
- `helper` k=385 (t=40.2 s): hand holding the bowl by its rim, held=bowl 0.43: correct.
- `leader` k=192 (t=20.9 s): held=bowl for both hands, but both detected hands enter from the LEFT of the frame, i.e.
  they are the partner's hands (the man stands at the left of the leader's view, mixing in the bowl); the leader's own
  right hand at the bottom right, holding a tool, is not detected at all. Misattribution: the contact is real, the
  person is wrong.
- `leader` k=290 (t=30.7 s): L held=cutting board (fingertips resting on the board's edge, marginal but defensible);
  R held=bowl: the hand in the bowl on the stove again enters from the left, most likely the partner's.

So on the leader stream a good share of "contacts" belong to the partner: MediaPipe with `num_hands=2` returns whichever
two hands it finds, and the `hands` stage does not separate the wearer's hands from the partner's. The handedness label
is MediaPipe's, which assumes a mirrored image; on an unmirrored ego frame L/R may be swapped (k=385: a hand on the
right side of the frame is labelled "Left").

### Against the CoMind handover annotations

`data/raw/comind/annotations/dataset_handover_consolidated.json`, recording `43276420-...`. The leader clip starts at
source time **590.0 s** (frame-matched: clip frames at 8/22/40 s match source frames at 598/612/630 s with mean abs
diff 0.8-1.0 vs median 28 over a 52 s search window; the render script's default of 596 s is not the clip offset).
Assuming the annotation times are on the `leader_trimmed_sync` timeline, `t_ref = t_annotation - 590`. Three annotated
handovers fall inside the common window (1.67-48.27 s):

| annotation | object | flow | t_ref | contact events nearby |
|---|---|---|---|---|
| 018240 | bowl | leader -> helper | 18.00-18.40 | helper R grasp "pan" (the bowl) at 17.07, held until 19.87; no leader release near 18 s (leader has no contact between 10.3 and 20.3 s) |
| 018260 | spoon | leader -> helper | 18.67-19.53 | nothing: "spoon" is never a held label in either stream (YOLO-World finds no spoon in the leader's hand) |
| 018941 | bowl | helper -> leader | 41.37-43.13 | helper L release bowl at 40.77, leader R grasp bowl at 41.57, release 42.07; helper L grasp bowl 43.57 |

One of three handovers is recovered as giver release + receiver grasp of the right object within ~0.6 s of the annotated
start; one gives only the receiver's grasp (0.9 s early, object class wrong); one is missed (object not detected).

## Tests

`tests/test_playground_contact.py` (6 tests): the containment / tie-break / enlargement / palm-only rules; majority
vote and hysteresis timings (start 3 frames, end 5 frames, back-fill, object switch); a synthetic stream with a 2 s hold,
a 2-frame glitch absorbed, and a 0.4 s second-hand contact; the stage end to end on a tmp episode (npz, events with
reference times and person, records_extra columns); the ZED depth check rejecting a candidate whose box depth is 40 cm
from the hand; skip without inputs.

## Not verified / limits

- Depth agreement never ran on real data (no ZED depth in the clip).
- No per-frame ground truth for held objects; the numbers above are self-consistency plus 4 inspected frames and 3
  annotated handovers.
- Wearer vs partner hands are not separated; on the leader stream this is the main error source.
- `export` was not re-run (existing stage); `records_extra.parquet` was checked for shape (466 x 8) and values only.
