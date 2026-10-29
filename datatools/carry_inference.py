"""Project shared intervals onto inference observations, preserving canonical states."""
from datatools.carry_policy import reconstruct_spells, STATE_CONTRACT


def populate(plan, match, controls, canonical, policy):
    audit = reconstruct_spells(controls, canonical, match.tracking, match.fps, policy,
                              independent_receipts=plan.provenance["state_contract"] == STATE_CONTRACT)
    for record in audit.to_dict("records"):
        valid = record["valid"]
        owner, period = record["carrier_id"], record["period_id"]
        start = record["start_frame"] if valid else None
        end = record["terminal_frame"] if valid else None
        receiver = record["receiver_id"]
        endpoint = receiver if isinstance(receiver, str) else owner
        start_id = plan.observe(period, start, owner, "start", record["start_event_id"]) if valid else None
        end_id = plan.observe(period, end, endpoint, "end", record["terminal_event_id"]) if valid else None
        spell = dict(spell_id=len(plan.spells) + 1, period_id=period, owner_id=owner,
            start_frame=start, end_frame=end, start_event_id=record["start_event_id"],
            terminal_event_id=record["terminal_event_id"], start_kind=record["start_kind"],
            end_kind=record["terminal_kind"], start_state_id=start_id, end_state_id=end_id,
            supported_until=end, break_after=not valid, reason=record["reason"],
            valid=valid, next_spell_id=None, next_state_id=None)
        plan.spells.append(spell)
        if not valid:
            plan.diagnostics.append(dict(event_id=record["start_event_id"], reason=record["reason"],
                                         carry_id=record["carry_id"]))
    records = audit.to_dict("records")
    for i, spell in enumerate(plan.spells):
        following = plan.spells[i+1] if i+1 < len(plan.spells) else None
        if (spell["valid"] and following and following["valid"]
                and spell["period_id"] == following["period_id"]
                and records[i].get("chain", 0) == records[i+1].get("chain", 0)):
            spell.update(next_spell_id=following["spell_id"], next_state_id=following["start_state_id"])
        else:
            spell["break_after"] = True
    by_terminal = {spell["terminal_event_id"]: spell for spell in plan.spells}
    for action in plan.actions:
        spell = by_terminal.get(action["original_event_id"])
        if spell and spell["owner_id"] == action["owner_id"] and spell["valid"]:
            action.update(spell_id=spell["spell_id"], start_state_id=spell["start_state_id"],
                          end_state_id=spell["end_state_id"], next_state_id=spell["next_state_id"])
    return plan
