"""Preserve physical self-edge inputs independently of learned node masks."""
import torch

from datatools import config
from models.edge_schema import edge_names, mask_edge_inputs

SELF_EDGE_ATTR = "node_self_edge_attr"


def original_self_attributes(graph):
    """Use the first existing loop; reconstruct only missing rows from raw inputs."""
    width = graph.edge_attr.shape[1]
    names = edge_names(graph)
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
    if "velocity_angle_cos" in names and missing.any():
        if graph.x is None or graph.x.shape[1] <= config.NODE_FEATURE_VY:
            raise ValueError("Missing self-edge velocity attributes require unmasked vx/vy node inputs.")
        velocity = graph.x[missing, config.NODE_FEATURE_VX:config.NODE_FEATURE_VY + 1]
        if not torch.isfinite(velocity).all():
            raise ValueError("Cannot reconstruct self-edge angles from nonfinite velocity.")
        squared = velocity.square().sum(-1)
        speed = squared.sqrt().clamp_min(1e-6)
        # Intentionally identical to existing graph generation, including zero speed.
        angle = torch.acos((squared / (speed * speed)).clamp(-1, 1))
        result[missing, names.index("velocity_angle_cos")] = torch.cos(angle)
        result[missing, names.index("velocity_angle_sin")] = torch.sin(angle)
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
    attributes = mask_edge_inputs(attributes, index, graph.x, args, names=edge_names(graph),
                                  possessor_mask=getattr(graph, "inference_possessor_mask", None))
    return index, attributes
