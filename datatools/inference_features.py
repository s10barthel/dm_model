"""Feature requirements for temporary inference graphs; persisted graphs stay full."""
from models.feature_masks import NODE_GROUPS, node_permissions
from models.input_scaling import node_schema


EXTENDED_FEATURES = NODE_GROUPS["extended-node-features"]


def runtime_graph_feature_kwargs(models):
    """Compute the union of retained features, including separately loaded auxiliaries.

    Unknown layouts and raw-graph cache consumers retain historical computation.
    Callers generating saved graphs/caches must not opt into this runtime policy.
    """
    from physical_pass_model import model_uses_physical_xpass

    needed = set()
    consumers = list(models.values())
    for model in list(consumers):
        consumers.extend(getattr(model, "_xpass_auxiliary_models", {}).values())
    for model in consumers:
        args = getattr(model, "args", None)
        if not isinstance(args, dict) or not args.get("node_in_dim"):
            return {}
        if (args.get("lane_survival", False) or model_uses_physical_xpass(args)
                or args.get("task") in {"success_intent", "failure_receiver"}
                or args.get("feature_variant") == "success_intent"):
            return {}
        schema = node_schema(args)
        # Custom or unrecognized layouts must not reinterpret fixed-width slots.
        expected = {"node." + name: index for index, name in enumerate(EXTENDED_FEATURES, 19)}
        if any(schema.get(name) != index for name, index in expected.items()):
            return {}
        permissions = node_permissions(args, schema)
        needed.update(name for name in EXTENDED_FEATURES if permissions["node." + name])
    if needed == set(EXTENDED_FEATURES):
        return {}
    return {"computed_extended_features": tuple(name for name in EXTENDED_FEATURES if name in needed)}
