"""Training projection of shared Sportec possession intervals."""
import pandas as pd

from datatools.carry_policy import DEFINITION, reconstruct_spells, tracking_table


def derive(control_events, canonical_events, tracking, fps=25, *, policy=None):
    from datatools.ball_carries import _canonical_return_index
    tracking = tracking_table(tracking)
    canonical = canonical_events.reset_index(drop=True)
    audit = reconstruct_spells(control_events, canonical, tracking, fps, policy)
    columns = ["action_key", "carry_definition", "carry_id", "segment_id", "period_id", "frame_id",
               "receive_frame_id", "object_id", "receiver_id", "spadl_type", "action_type", "success",
               "duration", "start_x", "start_y", "end_x", "end_y", "start_event_id", "terminal_event_id",
               "terminal_kind", "return_event_index"]
    segments = []
    audit["retained"] = False
    audit["return_event_index"] = pd.Series(index=audit.index, dtype="Int64")
    for i, spell in audit.iterrows():
        return_index = _canonical_return_index(canonical, int(spell.period_id), int(spell.terminal_frame), spell.terminal_event_id)
        audit.at[i, "return_event_index"] = return_index
        if not spell.valid:
            continue
        count = (int(spell.terminal_frame) - int(spell.start_frame)) // fps
        if count < 1 or return_index is None:
            audit.at[i, "reason"] = "short_or_missing_return"
            continue
        audit.at[i, "retained"] = True
        for segment_id in range(int(count)):
            start = int(spell.start_frame) + segment_id * fps
            end = int(spell.terminal_frame) if segment_id == count - 1 else start + fps
            receiver = spell.receiver_id if isinstance(spell.receiver_id, str) else spell.carrier_id
            segments.append(dict(action_key=f"carry:{int(spell.carry_id)}:{segment_id}", carry_definition=DEFINITION,
                carry_id=int(spell.carry_id), segment_id=segment_id, period_id=int(spell.period_id),
                frame_id=start, receive_frame_id=end, object_id=spell.carrier_id, receiver_id=receiver,
                spadl_type="ball_carry", action_type="dribble", success=bool(spell.success) if segment_id == count - 1 else True,
                duration=(end-start)/fps, start_x=tracking.at[start, "ball_x"], start_y=tracking.at[start, "ball_y"],
                end_x=tracking.at[end, "ball_x"], end_y=tracking.at[end, "ball_y"],
                start_event_id=spell.start_event_id, terminal_event_id=spell.terminal_event_id,
                terminal_kind=spell.terminal_kind, return_event_index=return_index))
    result = pd.DataFrame(segments, columns=columns)
    result.attrs["carry_definition"] = DEFINITION
    audit.attrs["carry_definition"] = DEFINITION
    return result, audit
