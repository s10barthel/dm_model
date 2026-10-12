"""Architecture selection and saved model identity (not dataset configuration)."""
ARCHITECTURES = ("gat", "transformer", "relationship", "relationship-attention")
GNN_ARCHITECTURES = (*ARCHITECTURES, "gcn", "gin")


def add_architecture_arguments(parser):
    parser.add_argument("--gnn-architecture", choices=ARCHITECTURES, default=None,
                        help="GNN encoder architecture; fresh training defaults to gat.")
    parser.add_argument("--model", default=None, help="Compatibility alias for architecture selection.")


def resolve_architecture(args):
    selected, old = args.get("gnn_architecture"), args.get("model")
    if selected is not None and old is not None and selected != old:
        raise ValueError("Conflicting --gnn-architecture and --model selections.")
    return selected or old or "gat"


def consumes_edge_attributes(args):
    return resolve_architecture(args) in ARCHITECTURES


def normalize_architecture(args, *, fresh=False):
    name = resolve_architecture(args)
    if name not in GNN_ARCHITECTURES:
        raise ValueError(f"Unsupported GNN architecture: {name}")
    args["model"] = args["gnn_architecture"] = name
    if args.get("gnn_architecture_version") is None:
        args["gnn_architecture_version"] = 1 if fresh or name not in ("gat", "gcn", "gin") else 0
    if args["gnn_architecture_version"] not in (0, 1):
        raise ValueError("Unsupported GNN architecture version.")
    if args.get("self_loop_policy") is None:
        args["self_loop_policy"] = "explicit_v1" if fresh and name in ARCHITECTURES else "legacy"
    if args["self_loop_policy"] not in ("legacy", "explicit_v1"):
        raise ValueError("Unsupported self-loop policy.")
    return args


def architecture_identity(args):
    name = resolve_architecture(args)
    if name not in GNN_ARCHITECTURES:
        return {"model": name}
    resolved = normalize_architecture(dict(args))
    return {key: resolved[key] for key in
            ("gnn_architecture", "gnn_architecture_version", "self_loop_policy")}
