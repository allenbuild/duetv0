# Playground stage `speech`: transcript + who spoke (2026-09-26)

`src/duet/playground/speech.py`. Turns the audio of an episode into one de-duplicated, speaker-attributed
transcript in reference time plus per-frame speech columns for `records.parquet`.

## The problem

Every camera in the room hears everyone. Transcribing each stream separately gives N copies of every
utterance (four on the CoMind clip), with different wordings, and no speaker labels. The wearer's own voice is
however by far the loudest thing in their own ego mic, which gives us attribution for free.

## Method

1. **Audio.** For every stream with audio, ffmpeg extracts 16 kHz mono pcm16 for the common window:
   stream time `common_start_s - offset_s` for `common_end_s - common_start_s` seconds
   (`derived/speech/<stream>.wav`). All wavs therefore start at `common_start_s` in reference time, so wav
   time `w` is `t_ref = common_start_s + w` for every stream.
2. **Transcription.** faster-whisper (`small` by default, `DUET_WHISPER_MODEL` overrides; CPU, `int8`,
   `word_timestamps=True`, `vad_filter=True`, `beam_size=5`, language auto). Model files are cached in
   `data/playground/models/`. The audio is **peak-normalised to 0.9 FS** before decoding: the Aria leader mic on
   the CoMind clip is -42 dB RMS and without normalisation whisper decoded it into two lower-case run-on
   segments spanning 39 s; normalised it gives 13 clean punctuated segments.
   Output `derived/speech/<stream>.json`: segments with words, all times in reference seconds.
3. **Attribution.** For every segment (from any stream) compute the RMS in dB of each **ego** wav in the
   segment window minus that mic's own median frame level (100 ms frames over the whole wav). Speaker =
   argmax. Relative-to-median removes per-mic gain (the two Aria mics differ by ~5 dB in floor). Margin over the
   runner-up is stored as `speaker_margin_db`. Without ego mics the speaker is `unknown`.
4. **De-duplication** (`dedup_segments`). Visit segments best-first (heard by the speaker's own ego mic, then
   mean word probability, then length) and keep each unless it duplicates a kept one:
   - rule 1: time overlap >= 50 % of the shorter AND >= 60 % of the shorter text's words matched
     (`difflib` on word lists, relative to the shorter text, so a fragment or a run-on that swallowed several
     utterances matches);
   - rule 2: same attributed speaker, overlap >= 80 % of the shorter, both <= 8 s: one mouth cannot say two
     things at once, so differing words are two mics' transcriptions of one utterance. Different speakers at
     the same time (cross-talk) are kept; run-ons longer than 8 s only match through rule 1.
   `derived/speech/transcript.json` holds the kept segments (with `source` stream, `rel_db`, `speaker_margin_db`)
   and the dropped ones with `duplicate_of`.
5. **Per-frame columns** `derived/speech/records_extra.parquet`, one row per common-timeline frame
   (`derived/frames/<stream>` count, or `duration * proc_fps` if frames are not extracted): `speaking_<person>`
   (a kept segment of that person covers the frame time), `words_last_3s_<person>` (that person's words ending in
   `(t-3, t]`; a wordless segment counts as one word at its end), `any_speech`. `export` merges them as
   `speech_speaking_leader`, `speech_words_last_3s_helper`, `speech_any_speech`, ...

Status detail example: `whisper-small on 4 streams (en): 16 utterances / 86 words (dropped 33 duplicates); by speaker leader 6, helper 10`.

## Result on `comind_43276420_clip` (two Aria ego mics + two GoPros, 46.6 s common window, ~30 s wall time)

49 raw segments across the four streams -> 16 kept. Every kept utterance but one comes from the speaker's own
ego mic.

```
 1.67- 4.89 leader  (+2.9 dB)  to add it, because it's too... just... there.
 6.15- 7.27 helper  (+13.8 dB) Okay, but no more carrot?
 8.07- 9.35 leader  (+2.1 dB)  No, with the curry, it's fine.
14.35-15.79 leader  (+3.8 dB)  Do you want to add it?
16.17-16.51 helper  (+4.9 dB)  Sure.
17.67-20.05 helper  (+7.8 dB)  Wait, I think it's still a bit floury.
26.23-27.15 helper  (+19.1 dB) So I mix it in?
27.43-27.65 leader  (+6.0 dB)  Mm-hmm.
28.27-29.31 leader  (+1.4 dB)  We can mix it there.
29.29-30.79 helper  (+5.5 dB)  It might be a bit tricky
31.39-32.27 helper  (+9.1 dB)  because it's thicker.
31.92-32.42 helper  (+16.3 dB) Just be careful.            <- gopro_front only; possibly a mis-hearing of the line above
38.63-40.49 leader  (+3.0 dB)  Well, it's a special recipe.
40.98-43.00 helper  (+6.4 dB)  Yeah, we might need to mix it back in the bowl.
43.62-43.98 helper  (+8.5 dB)  I'm not sure.
44.80-47.74 helper  (+0.9 dB)  I think it might be important to do that.
```

Per-frame: leader speaking 20 % of frames, helper 27 %, any speech 47 %.

**Quality.** Whisper-small is usable but not clean: the first utterance is clipped by the clip start; the
same word is heard as "carrot" (helper mic, twice), "curry" (leader mic) and "hair" (GoPros), and the GoPro
copies are generally worse ("They want the most volume" for "Do you want to add it?"). The kept copy is the
own-mic one, which is the best of the four in every case we checked by reading the alternatives. "Just be
careful" at 31.9 s is only heard by gopro_front and overlaps "because it's thicker" by 70 % (below the 80 %
rule-2 threshold), so it survives; it may be a duplicate.

**Attribution plausibility.** The turn structure reads like a real dialogue (question -> answer alternates
between the two labels: "Do you want to add it?" leader / "Sure." helper; "So I mix it in?" helper / "Mm-hmm."
leader). Margins are asymmetric: helper utterances win by 5-19 dB, leader utterances by only 1.4-3.8 dB. The
leader mic has a 5 dB higher noise floor and lower peak level, so the leader's voice stands out less above its
own median. Before normalisation whisper on the leader mic transcribed ONLY the leader's lines, which is
independent evidence that the mic favours its wearer. The 44.8 s utterance (margin 0.9 dB) is the one we would
not bet on. No ground truth: nobody has listened to the clip and labelled speakers, so this is plausibility,
not accuracy.

## Tests

`tests/test_playground_speech.py`: attribution on synthetic mics where the louder-gain mic would win on raw
RMS but the wearer wins relative to the median; de-dup rules (identical text across three mics keeps the own-mic
copy; same speaker + same time + different words merges; cross-talk kept; fragment inside a same-speaker
utterance dropped; run-on swallowing three utterances dropped by text; two long run-ons not merged by time);
per-frame columns; and two slow tests (`DUET_SLOW_TESTS=1`): whisper on ffmpeg-generated speech-free pink
noise (proves extraction, model load, empty transcript, 60 rows of columns, and that `export` merges
`speech_*` columns), and 5 s of the CoMind clip (segments inside the window, speakers in {leader, helper}).

## Limits / next

- No diarisation inside a segment: a segment gets one speaker; whisper's VAD segmentation decides the unit.
- Attribution needs one ego mic per person; with exo mics only everything is `unknown`.
- Text similarity is English-word based (`[a-z0-9']+`); other scripts fall back to rule 2 only.
- Whisper `small` int8 on CPU: ~7 s per 47 s stream. `medium` would fix some of the mis-hearings at ~4x cost.
- The `probe` `has_audio` flag gates extraction; streams whose audio is silent still get transcribed
  (VAD then returns nothing).
