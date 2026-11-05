"""The xT return definitions shared by generation and targeted refresh."""
from datatools import config, utils
from project_config import parse_return_type


def label_xt_for_return_type(events, return_type):
    kind, value, skip_first = parse_return_type(return_type)
    eligible = tuple(config.XT_ACTION_TYPES)
    if kind == "extended_possession":
        from datatools.possession_returns import label_extended_possession_events
        return label_extended_possession_events(events, {"xt"}, max_actions=value)
    if kind == "next_plus":
        from datatools.return_windows import action_xt_windows
        selected = events.copy()
        pairs, _ = action_xt_windows(events, events, return_type)
        selected[["scores_xT", "concedes_xT"]] = pairs
        return selected
    if kind == "next":
        return utils.label_xt_returns(events, lookahead_len=int(value), eligible_types=eligible, skip_first=skip_first)
    if kind == "in":
        return utils.label_xt_in_state_returns(events, action_offset=int(value), eligible_types=eligible)
    if kind == "disc_poly_max":
        b, z = value
        return utils.label_polynomial_xt_returns(events, b=b, z=z, eligible_types=eligible,
                                                stop_at_set_pieces=return_type.endswith("_spstop"))
    return utils.label_discounted_xt_returns(events, gamma=float(value), eligible_types=eligible,
                                             skip_first=skip_first,
                                             aggregation="max" if kind == "disc_max" else "probability")
