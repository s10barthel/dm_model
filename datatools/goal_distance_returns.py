"""Goal-distance return selection shared by generation and offline refresh."""
from datatools import config, utils
from project_config import parse_return_type


def label_goal_distance_for_return_type(events, return_type):
    kind, value, skip_first = parse_return_type(return_type)
    eligible = tuple(config.XT_ACTION_TYPES)
    if kind == "next":
        return utils.label_goal_distance_returns(events, lookahead_len=int(value),
                                                eligible_types=eligible, skip_first=skip_first)
    if kind == "in":
        return utils.label_goal_distance_in_state_returns(events, action_offset=int(value),
                                                         eligible_types=eligible)
    if kind == "disc_poly_max":
        b, z = value
        return utils.label_polynomial_goal_distance_returns(
            events, b=b, z=z, eligible_types=eligible, stop_at_set_pieces=return_type.endswith("_spstop"))
    return utils.label_discounted_goal_distance_returns(
        events, gamma=float(value), eligible_types=eligible, skip_first=skip_first,
        aggregation="max" if kind == "disc_max" else "probability")
