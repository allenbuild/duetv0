"""Read only the handover fields verified in ``comind_format.md``.

The inspected release uses a per-recording object keyed by opaque segment keys.
File loading accepts that verified layout by default and retains the keys.
An explicit selector supports other independently verified layouts; this module
does not guess wrapper keys or recursively search for annotations. Annotation seconds
belong to an isolated clock domain whose origin and relation to other CoMind
modalities remain unknown. Source labels never establish participant identity.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import TypeAlias, cast
from uuid import UUID

from duet.schemas.common import Provenance
from duet.schemas.episode import HandoverAnnotation, JSONValue
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit

SegmentSelector: TypeAlias = Callable[[JSONValue], Sequence[Mapping[str, JSONValue]]]

_OPAQUE_FIELDS = (
    "bbox",
    "transcript_10s",
    "numeric_id",
    "description",
    "annot_type",
    "skip",
)


def _validate_uuid(recording_uuid: str) -> None:
    if not isinstance(recording_uuid, str):
        raise TypeError("recording_uuid must be a UUID string")
    try:
        UUID(recording_uuid)
    except ValueError as error:
        raise ValueError("recording_uuid must be a valid UUID") from error


def _optional_frame(segment: Mapping[str, JSONValue], key: str) -> int | None:
    value = segment.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{key} must be an integer frame index or null")
    return value


def _optional_time(segment: Mapping[str, JSONValue], key: str) -> int | float | Decimal | None:
    value = segment.get(key)
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float, Decimal))
    ):
        raise ValueError(f"{key} must be a numeric timestamp in seconds or null")
    return value


def _optional_label(segment: Mapping[str, JSONValue], key: str) -> str | None:
    value = segment.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be an uninterpreted string label or null")
    return value


def parse_handover_segments(
    segments: Sequence[Mapping[str, JSONValue]],
    *,
    recording_uuid: str,
    provenance: Provenance,
) -> tuple[HandoverAnnotation, ...]:
    """Convert an explicitly selected sequence of verified handover segments.

    Only entries with a literal JSON ``false`` skip value are retained. Missing
    boundaries remain missing; no frame-rate conversion or clock mapping is
    inferred. Categories and initiation types preserve their source structure,
    including scalar/list distinctions, without assigning extra semantics.
    """
    _validate_uuid(recording_uuid)
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)):
        raise TypeError("segments must be an explicitly selected sequence of mappings")
    clock_domain = ClockDomain(f"comind:{recording_uuid}:handover_annotations")
    annotations = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise TypeError(f"handover segment {index} must be a mapping")
        if segment.get("skip") is not False:
            continue
        segment_provenance = Provenance(
            source=provenance.source,
            detail=f"recording {recording_uuid}, selected handover segment {index}",
            parents=(provenance,),
        )
        annotations.append(
            HandoverAnnotation(
                recording_id=recording_uuid,
                annotation_id=None,
                start_frame=_optional_frame(segment, "start_frame"),
                end_frame=_optional_frame(segment, "end_frame"),
                start_time=Timestamp(
                    raw_value=_optional_time(segment, "start_time"),
                    unit=TimeUnit.SECONDS,
                    clock_domain=clock_domain,
                    provenance=segment_provenance,
                ),
                end_time=Timestamp(
                    raw_value=_optional_time(segment, "end_time"),
                    unit=TimeUnit.SECONDS,
                    clock_domain=clock_domain,
                    provenance=segment_provenance,
                ),
                initiator=_optional_label(segment, "initiator"),
                delivering_flow=_optional_label(segment, "delivering_flow"),
                initiation_type=segment.get("initiation_type"),
                object_category_level_1=segment.get("object_category_level_1"),
                object_category_level_2=segment.get("object_category_level_2"),
                object_category_level_3=segment.get("object_category_level_3"),
                provenance=segment_provenance,
                source_fields={key: segment[key] for key in _OPAQUE_FIELDS if key in segment},
            )
        )
    return tuple(annotations)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON numeric constant {value!r} is not supported")


def _unique_object(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_handover_recording(
    recording: JSONValue,
    *,
    recording_uuid: str,
    provenance: Provenance,
) -> tuple[HandoverAnnotation, ...]:
    """Parse the inspected native recording object, preserving its source keys.

    Keys become opaque annotation identifiers. Their apparent relationship to
    start frames is not used to manufacture or change any timestamp or frame.
    Source object order is retained, even when it is not chronological.
    """
    _validate_uuid(recording_uuid)
    if not isinstance(recording, Mapping):
        raise TypeError("native recording annotations must be a segment-keyed object")
    annotations = []
    for key, segment in recording.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("native annotation keys must be nonempty strings")
        if not isinstance(segment, Mapping):
            raise TypeError("native recording annotation values must be segment objects")
        segment_provenance = Provenance(
            source=provenance.source,
            detail=f"recording {recording_uuid}, source annotation key {key!r}",
            parents=(provenance,),
        )
        parsed = parse_handover_segments(
            [segment], recording_uuid=recording_uuid, provenance=segment_provenance
        )
        annotations.extend(replace(annotation, annotation_id=key) for annotation in parsed)
    return tuple(annotations)


def load_handover_annotations(
    path: str | Path,
    *,
    recording_uuid: str,
    segment_selector: SegmentSelector | None = None,
) -> tuple[HandoverAnnotation, ...]:
    """Load a recording from ``dataset_handover_consolidated.json``.

    ``path`` is caller supplied and has no implicit dataset location. The
    verified root ``data`` dictionary is indexed by the exact recording UUID.
    By default the inspected segment-keyed recording object is parsed and its
    keys retained. An optional ``segment_selector`` receives the recording's
    JSON value and returns segments using other independently verified layout
    knowledge; selected sequences have no inferred source annotation keys.
    Decimal JSON numbers retain their precision; no timestamp conversion is
    made. Invalid or ambiguous JSON objects raise rather than being repaired.
    """
    _validate_uuid(recording_uuid)
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        document = json.load(
            stream,
            parse_float=Decimal,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    if not isinstance(document, Mapping):
        raise TypeError("handover annotation document must be a JSON object")
    data = document.get("data")
    if not isinstance(data, Mapping):
        raise TypeError("handover annotation document must contain a 'data' UUID dictionary")
    if recording_uuid not in data:
        raise KeyError(f"recording UUID {recording_uuid!r} not present in annotation data")
    recording = cast(JSONValue, data[recording_uuid])
    provenance = Provenance(source=str(path), detail="CoMind handover annotations")
    if segment_selector is None:
        return parse_handover_recording(
            recording, recording_uuid=recording_uuid, provenance=provenance
        )
    segments = segment_selector(recording)
    return parse_handover_segments(
        segments,
        recording_uuid=recording_uuid,
        provenance=provenance,
    )
