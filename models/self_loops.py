"""Preserve physical self-edge inputs independently of learned node masks."""
import torch

from datatools import config
from models.edge_feature_config import normalize_v_edge_feature_args

SELF_EDGE_ATTR = "node_self_edge_attr"


def original_self_attributes(graph):
    """Use the first existing loop; reconstruct only missing rows from raw inputs."""
    width = graph.edge_attr.shape[1]
    if width not in (2, 4, 5):
        raise ValueError(f"Self-loop reconstruction requires edge width 2, 4 or 5; found {width}.")
    count = graph.num_nodes
    saved = getattr(graph, SELF_EDGE_ATTR, None)
    if saved is not None:
        if saved.shape[0] != count or saved.shape[1] < width:
            raise ValueError("Self-edge metadata does not match the graph schema.")
        return saved[:, :width].clone()
    result = graph.edge_attr.new_zeros((count, width))
    result[:, 1] = 1
    sender, receiver = graph.edge_index
    positions = torch.arange(sender.numel(), device=sender.device)
    loop = sender == receiver
    first = torch.full((count,), sender.numel(), dtype=torch.long, device=sender.device)
    first.scatter_reduce_(0, sender[loop], positions[loop], reduce="amin", include_self=True)
    exists = first < sender.numel()
    result[exists] = graph.edge_attr[first[exists]]
    missing = ~exists
    if width >= 4 and missing.any():
        if graph.x is None or graph.x.shape[1] <= config.NODE_FEATURE_VY:
            raise ValueError("Missing self-edge velocity attributes require unmasked vx/vy node inputs.")
        velocity = graph.x[missing, config.NODE_FEATURE_VX:config.NODE_FEATURE_VY + 1]
        if not torch.isfinite(velocity).all():
            raise ValueError("Cannot reconstruct self-edge angles from nonfinite velocity.")
        squared = velocity.square().sum(-1)
        speed = squared.sqrt().clamp_min(1e-6)
        # Intentionally identical to existing graph generation, including zero speed.
        angle = torch.acos((squared / (speed * speed)).clamp(-1, 1))
        result[missing, 2] = torch.cos(angle)
        result[missing, 3] = torch.sin(angle)
    return result


def capture_self_attributes(graph):
    setattr(graph, SELF_EDGE_ATTR, original_self_attributes(graph))
    return graph


def explicit_self_edges(graph, args):
    """Return local edge tensors with exactly one loop and training masks applied."""
    attributes = original_self_attributes(graph)
    index = graph.edge_index
    other = index[0] != index[1]
    nodes = torch.arange(graph.num_nodes, device=index.device)
    index = torch.cat((index[:, other], torch.stack((nodes, nodes))), dim=1)
    attributes = torch.cat((graph.edge_attr[other], attributes), dim=0)
    mask_args = dict(args)
    mask_args.setdefault("edge_in_dim", attributes.shape[1])
    settings = normalize_v_edge_feature_args(mask_args)
    incident = None
    if "no_poss" in (settings["v_edge_feature_mode"], settings["relative_speed_edge_feature_mode"]):
        possessor = getattr(graph, "inference_possessor_mask", None)
        if possessor is None:
            if graph.x.shape[1] <= config.NODE_FEATURE_IS_POSSESSOR:
                raise ValueError("Possessor edge masking requires the possessor role or preserved role metadata.")
            possessor = graph.x[:, config.NODE_FEATURE_IS_POSSESSOR] == 1
        incident = possessor[index[0]] | possessor[index[1]]
    for columns, mode in ((slice(2, 4), settings["v_edge_feature_mode"]),
                          (slice(4, 5), settings["relative_speed_edge_feature_mode"])):
        if mode == "none":
            attributes[:, columns] = 0
        elif mode == "no_poss":
            attributes[incident, columns] = 0
    return index, attributes
