# Dorm recording rig v0 — hardware spec and CAD brief

Purpose: the first station for recording two people doing coordination tasks at a table, producing footage the playground processes end to end (board calibration → tags → head pose → triangulated bodies → export). Two flavours of head camera run on the same station: the ZED Mini (reference quality, tethered) and the cheap body-cam (deployable, untethered). Everything below is what the CAD person needs to print and what to buy.

## 1. Station layout

```
                 wall with tags 20, 21 (12 cm) at 1.6 m height, 1.5 m apart
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
   exoA ●  (tripod, 2.0 m high, 2.5 m from table centre)       │
        │            ┌────────────────────┐                    │
        │   person L │      table         │ person H           │
        │   (cap 1)  │  120–180 × 60–80cm │ (cap 2)            │
        │            └────────────────────┘                    │
        │      ChArUco board lies flat here at setup (10 s)    │
        │                                          ● exoB      │
        └──────────────────────────────────────────────────────┘
   exoA and exoB 70–90° apart as seen from the table centre; each sees both people,
   both cap tags, and the board at setup. Laptop for the ZED under the table edge.
```

Lighting: at least 500 lux on the table (two cheap LED panels or bright desk lamps pointed up at the ceiling). Cameras must expose at 1/120 s or faster, or hands smear. No windows behind people.

## 2. Cameras and settings

| role | device | setting | notes |
|---|---|---|---|
| head (reference) | ZED Mini | 720p at 30 fps, SVO or left.mp4 | 124.5 × 30.5 × 26.5 mm, 60 g, USB-C 3.0, 1.5 m cable (4 m certified extension available), FOV 102° H, baseline 63 mm |
| head (deployable) | body-cam clone, 1080p30, ≥ 120° FOV, mic, ≥ 8 h battery, < 100 g | 1080p at 30 fps | records to microSD, no tether |
| fixed exoA / exoB | any 1080p30 or 4K30 camera with a 1/4"-20 tripod socket, or two more body-cams | 4K30 if available, else 1080p30 | 4K makes the 8 cm cap tags decodable at 3 m; at 1080p keep the cams within 2.5 m |
| board | ChArUco 7 × 5, 50 mm squares, 37 mm markers, dictionary 5X5_100 | printed A3 at 100 % scale on matte paper, glued to 5 mm foam board, must be flat | this is the world frame |
| tags | AprilTag 36h11: cap tags id 1 (leader) and 2 (helper) at 8 cm; wall tags 20, 21 at 12 cm; optional object tags 10+ at 4 cm | printed matte, black border ≥ 1 tag-cell wide, glued flat to the printed plates below | ids match `rig.json` |
| audio | every camera records its own mic | clap twice at the start of every take | the aligner uses audio; no clap, no sync |

## 3. Parts to 3D print (the CAD brief)

Material PLA or PETG, 3 mm walls, matte finish. Every tag surface must be flat within 0.5 mm across its width, because pose from a bent tag is wrong by centimetres. Print tag plates face down on the bed.

### Part A — cap mount (make 2)
The universal head mount. Attaches to the front panel and brim of a stiff baseball cap.

- Brim clip: two spring jaws gripping the brim, 55–65 mm apart, brim thickness 3–5 mm, plus two 6 mm holes for M4 bolts through the cap's front panel as a backup.
- Front interface for the camera: a GoPro-style three-finger mount (fingers 3.0 mm thick, gaps 3.3 mm, M5 through-bolt) **and** a 1/4"-20 heat-set brass insert 10 mm behind it. Almost every body-cam ships with a GoPro or 1/4"-20 adapter, so one mount fits any camera we buy.
- Camera position: optical axis pointing 15° below horizontal when the wearer looks straight ahead (people look at the table), centred left–right, front face 20–30 mm ahead of the brim edge.
- Tag plate on top: 95 × 95 mm flat plate for the 8 cm tag, raised 1 mm rim to align the printed tag, tilted 30° forward from horizontal so the fixed cameras at 2 m height see it face-on. Emboss the tag id on the underside.
- Rigid link between tag plate and camera interface. The offset between tag centre and camera lens is the `T_tag_headcam` the pipeline needs; make it a fixed, documented dimension (nominal: tag centre 60 mm above and 20 mm behind the camera's front face). Report the final numbers from the CAD, we enter them in `rig.json`.
- Mass target under 60 g without camera. Cable exit at the back for the ZED, strain-relief slot.

### Part B — ZED Mini cradle (make 1)
Slides onto Part A's GoPro fingers.

- Cradle for the 124.5 × 30.5 × 26.5 mm body, 1.5 mm clearance, open front for the lenses (do not shadow the 102° field of view: nothing within 10 mm of either lens), open bottom for the USB-C plug.
- Fastening: the ZED Mini has 4 × M2 threaded holes (max screw depth 2.3 mm) and, on current revisions, a 1/4"-20 socket. Stereolabs publishes no CAD for the Mini, so **measure the hole pattern on our unit** before printing; if the 1/4"-20 socket exists use it, else the M2 holes.
- The lens-to-tag offset with the cradle mounted goes into `rig.json` as `T_tag_headcam` for the ZED stream.

### Part C — wall and object tag plates (make 4 wall, 6 object)
- Wall: 140 × 140 mm, 3 mm thick, 1 mm rim for a 12 cm tag, two keyhole slots for screws or Command strips.
- Object: 50 × 50 mm, 2 mm thick, rim for a 4 cm tag, flat back for double-sided tape. Only for objects we want tracked in 3D without depth; hands and detection work without them.

### Part D — board frame (make 1, optional)
- A3 (297 × 420 mm) frame with 8 mm lip, holds the printed ChArUco on foam board flat and protects the corners. Skip if the foam board stays flat on its own.

## 4. One-time measurements after printing

1. Mount the camera on Part A, photograph the assembly from the side against a ruler, and record tag-centre-to-lens offsets (x forward, y up, z right, in metres). Enter as `T_tag_headcam` per stream.
2. Intrinsics per fixed camera: wave the board slowly through each fixed camera's view for 30 s at setup once; the `calib` stage stores them in `rig.json`. Fixed cameras watching a static board cannot self-calibrate, this step is what makes triangulation metric.
3. Confirm 8 cm tag pixel size from each fixed camera at the seated head positions: needs ≥ 25 px across, else move the cameras closer or switch to 4K.

## 5. `rig.json` for this station

```json
{"board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037},
 "tag_size_m": 0.08,
 "head_tags": {"leader": 1, "helper": 2},
 "wall_tags": [20, 21],
 "object_tags": {},
 "T_tag_headcam": {"leader": "<from CAD, Part A>", "helper": "<from CAD, Part A>"},
 "hfov_deg": {"exoA": 120, "exoB": 120}}
```

## 6. Take protocol (2 minutes per take)

1. All cameras recording. Clap twice.
2. Lay the board flat on the table centre for 10 s so every fixed camera and both head cams see it. Remove it.
3. Both people sit; look at each camera for 2 s (tag check).
4. Run the task (handovers, two-person carry, hold-and-fasten). 3–10 minutes.
5. Clap twice, stop.

## 7. Shopping list (station, excluding computer)

| item | qty | approx |
|---|---|---|
| body-cam clones (2 head + 2 fixed) | 4 | $140–240 |
| tripods 2 m | 2 | $50 |
| stiff baseball caps | 2 | $20 |
| microSD 128 GB | 4 | $40 |
| LED light panels | 2 | $40 |
| A3 matte print + foam board, tag prints | 1 set | $15 |
| filament for Parts A–D | ~300 g | $10 |
| ZED Mini 4 m certified extension (optional) | 1 | $40 |
| total | | ≈ $300–450 |

## 8. Why these numbers

- 30 fps floor: a handover lasts ~0.3 s; 10 frames captures reach, grasp, release. We process at 10 fps sampled from 30.
- 8 cm cap tag: at 1080p with a 120° lens the camera resolves ~16 px per degree; 8 cm at 3 m is 1.5°, so ~25 px, the decode limit. 4K doubles the margin.
- 30° tag tilt: fixed cameras at 2.0 m looking down at seated heads (~1.2 m) see a horizontal tag at a grazing angle; tilting toward the camera keeps the four corners well separated.
- Board frame as world: every camera and tag is expressed relative to the board, so two people seen by different cameras land in one metric space.
