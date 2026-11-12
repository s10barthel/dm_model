"""Decomposition preserves expected-value outputs and exposes exact occurrence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from datatools.config import LABEL_COLUMNS, LABEL_INDEX
from datatools.xt_targets import training_target_definition
from models.outcome_decomposition import (decode, fit_definition, losses, mean_bce,
    validate_definition, head_metrics, distribution_diagnostics)
from models.outcome_learning import (normalize_learning_settings, expected_values,
    configure_outcome_learning, validate_learning_metadata, validate_state_support)


def metadata():
    return dict(xt_version="fixture", xt_fit_metadata_sha256="fit",
                xt_targets=dict(version=2, representation="absolute", clip_negative_gains=False))


def configuration(severity="scalar", task="outcome_conceding"):
    args = dict(task=task, learning_mode="decomposition", severity_mode=severity,
                use_xt=True, target_family="xt", xt_target_definition=training_target_definition(metadata()))
    if severity == "distributional":
        args["distribution_atoms"] = 7
    normalize_learning_settings(args)
    labels = torch.zeros(6, len(LABEL_COLUMNS))
    labels[:, LABEL_INDEX["success"]] = torch.tensor([0., 1., 0., 1., 0., 1.])
    for name in ("scores_xt", "concedes_xt"):
        labels[:, LABEL_INDEX[name]] = torch.tensor([0., 0., .001, .01, .2, 1.])
    args["outcome_decomposition_definition"] = fit_definition([labels], args, {"split": "training"})
    return args


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
def test_product_loss_selection_extremes_and_zero_positive_batches(severity):
    args = configuration(severity)
    width = 2 if severity == "scalar" else 1+args["outcome_decomposition_definition"]["severity_distribution"]["effective_atoms"]
    logits = torch.randn(4, 2, width, dtype=torch.double, requires_grad=True)
    pred = logits[torch.arange(4), torch.tensor([0, 1, 0, 1])]
    y = torch.tensor([0., .000001, .1, 1.], dtype=torch.double)
    weights = torch.tensor([1., .5, 2., 1.], dtype=torch.double)
    joint, occurrence, contribution = losses(pred, y, SimpleNamespace(**args), weights)
    torch.testing.assert_close(joint, occurrence+contribution)
    joint.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[0, 1]) == 0
    assert torch.count_nonzero(logits.grad[0, 0, 1:]) == 0
    decoded = decode(pred.detach(), args)
    torch.testing.assert_close(expected_values(pred.detach(), args),
        decoded["occurrence_probability"] * decoded["conditional_severity"])
    torch.testing.assert_close(mean_bce(pred.detach(), y, SimpleNamespace(**args), weights),
        (torch.nn.functional.binary_cross_entropy(decoded["prediction"], y, reduction="none")*weights).mean())
    zero_logits = torch.zeros(3, width, requires_grad=True)
    zero_loss, _, severity_loss = losses(zero_logits, torch.zeros(3), SimpleNamespace(**args))
    zero_loss.backward()
    assert severity_loss == 0 and torch.count_nonzero(zero_logits.grad[:,1:]) == 0
    extreme = torch.full((2, width), -10000., dtype=torch.double, requires_grad=True)
    extreme_loss = mean_bce(extreme, torch.tensor([1., 0.]), SimpleNamespace(**args))
    extreme_loss.backward()
    assert torch.isfinite(extreme_loss) and torch.isfinite(extreme.grad).all()
    forced = decode(torch.full((2, width), -float("inf")), args)
    assert (forced["prediction"] == 0).all() and torch.isnan(forced["conditional_severity"]).all()
    assert torch.isnan(decode(torch.full((2,width), float("nan")), args)["prediction"]).all()


@pytest.mark.parametrize("clipping", [True, False])
def test_gain_forbidden_even_when_no_negative_labels(clipping):
    gain = metadata(); gain["xt_targets"].update(representation="gain", clip_negative_gains=clipping)
    with pytest.raises(ValueError, match="absolute"):
        configure_outcome_learning(dict(task="outcome_scoring", learning_mode="decomposition", use_xt=True), gain)


def test_mode_validation_and_saved_contract():
    args = configuration("distributional")
    restored = dict(task=args["task"], learning_mode=None, use_xt=True)
    normalize_learning_settings(restored, args)
    assert restored["outcome_decomposition_definition"] == args["outcome_decomposition_definition"]
    for changes in ({"severity_mode":"scalar"}, {"distribution_atoms":9}, {"learning_mode":"scalar"}):
        with pytest.raises(ValueError):
            normalize_learning_settings({"task":args["task"], "use_xt":True, **changes}, args)
    with pytest.raises(ValueError):
        normalize_learning_settings(dict(task="pass_success", learning_mode="decomposition"))
    with pytest.raises(ValueError):
        configure_outcome_learning(dict(task=args["task"], learning_mode="decomposition", use_xt=False), {})
    with pytest.raises(ValueError):
        normalize_learning_settings(dict(task=args["task"], learning_mode="scalar", severity_mode="scalar"))
    broken = copy.deepcopy(args); broken["outcome_decomposition_definition"]["occurrence_rule"] = "epsilon"
    with pytest.raises(ValueError):
        validate_learning_metadata(broken,{})


def test_support_training_positives_rng_and_unseen_tiny_value():
    from models.outcome_learning import encode_targets
    rng = torch.get_rng_state().clone()
    args = configuration("distributional")
    assert torch.equal(rng,torch.get_rng_state())
    definition = args["outcome_decomposition_definition"]
    assert definition == configuration("distributional")["outcome_decomposition_definition"]
    support = definition["severity_distribution"]
    assert support["zero_label_count"] == 0 and support["training_label_count"] == 4
    atoms = torch.tensor(support["atoms"], dtype=torch.double)
    y = torch.tensor([1e-10], dtype=torch.double)
    torch.testing.assert_close((encode_targets(y,atoms)*atoms).sum(-1),y)
    args2=configuration(); labels=torch.zeros(4,len(LABEL_COLUMNS))
    labels[:,LABEL_INDEX['success']]=torch.tensor([0,1,0,1])
    with pytest.raises(ValueError,match="positives"):
        fit_definition([labels],args2,{})


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
@pytest.mark.parametrize("architecture", ["gat", "transformer", "relationship", "relationship-attention"])
def test_gnn_heads_and_state_support(severity, task, architecture):
    from models.gnn import GNN
    from test_benchmark_no_accel import make_model_args
    from test_goal_poss_dist import graph
    args = make_model_args(); args.update(configuration(severity,task))
    args.update(model=architecture, gnn_architecture=architecture, gnn_task="node_binary", out_dim=2, gnn_heads=2)
    model=GNN(args).eval(); batch=Batch.from_data_list([graph(),graph()])
    out=model(batch)
    assert out.shape[:2] == (batch.num_nodes,2)
    assert expected_values(out,args).shape == (batch.num_nodes,2)
    validate_state_support(model.state_dict(),args)
    out.sum().backward()
    assert model.decoder.nodewise_mlp[-1].weight.grad is not None
    if severity=="distributional":
        state=copy.deepcopy(model.state_dict()); state['decoder.outcome_atoms'][0]=.1
        with pytest.raises(ValueError):
            validate_state_support(state,args)


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
def test_real_epoch_evaluation_exports(tmp_path,severity):
    from models.utils import run_epoch
    from test import write_outcome_evaluation_artifacts
    import pandas as pd
    args=configuration(severity)
    args.update(gnn_task="node_binary",include_out=False,lambda_l1=0.,clip=10.,print_freq=99,
                export_outcome_decomposition=True,export_outcome_distributions=severity=="distributional",
                use_xg=False,use_epv=False)
    width=2 if severity=="scalar" else 1+args['outcome_decomposition_definition']['severity_distribution']['effective_atoms']
    samples=[]
    for i,y in enumerate([0.,0.,.001,.01,.2,1.]):
        x=torch.zeros(3,26);x[:,3]=torch.tensor([10.,40.,80.])
        graph=Data(x=x,edge_index=torch.tensor([[0,1],[1,0]]),edge_attr=torch.ones(2,2),
                   node_ids=['home_1','home_2','home_3'],evaluation_match_id='m',evaluation_source_row=i)
        label=torch.zeros(len(LABEL_COLUMNS));label[LABEL_INDEX['intent_index']]=i%3
        label[LABEL_INDEX['success']]=i%2;label[LABEL_INDEX['concedes_xt']]=y
        label[LABEL_INDEX['concedes_goal_next10']]=i%2
        samples.append((graph,label,torch.tensor(1.)))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
        def forward(self,graphs,destinations=None):
            return self.weight+graphs.x.new_zeros((graphs.num_nodes,2,width))
    metrics,evaluation=run_epoch(SimpleNamespace(**args),Model(),DataLoader(samples,batch_size=3),
        device='cpu',return_outcome_evaluation=True)
    assert metrics['ce_loss']==metrics['mean_soft_bce']
    write_outcome_evaluation_artifacts(tmp_path,model_id='outcome_conceding/fixture',task='outcome_conceding',
        outcome_evaluation=evaluation,outcome_bootstrap_resamples=0)
    reports=json.loads((tmp_path/'outcome_decomposition_metrics.json').read_text())
    assert reports['pooled_factual']['positives']==4
    assert reports['observed_failure']['positives']==2
    rows=pd.read_parquet(tmp_path/'outcome_decomposition.parquet')
    np.testing.assert_allclose(rows.prediction,rows.occurrence_probability*rows.conditional_severity)
    if severity=='distributional':
        assert 'outcome_crps' in reports['pooled_factual']


def test_one_class_and_empty_head_metrics():
    result=head_metrics([0,0],[.1,.2],[.3,.4])
    assert result['severity_mae'] is None and result['occurrence_roc_auc'] is None
    assert head_metrics([],[],[])['count']==0


@pytest.mark.parametrize("severity", ["scalar", "distributional"])
@pytest.mark.parametrize("task", ["outcome_scoring", "outcome_conceding"])
def test_inference_export_and_annotations(tmp_path,severity,task):
    import pandas as pd
    from inference import inference_gnn
    from models.outcome_decomposition import configure_exports,write_exports
    from models.outcome_learning import configure_distribution_exports,write_distribution_exports
    from scripts.outcome_decomposition_visualization import validate_models,annotate
    from test_pc_xpass_auxiliary import auxiliary_model
    from test_physical_xpass import make_physical_inference_match
    match=make_physical_inference_match([7,8],[1,1])
    for graph in match.graph_features_0:
        graph.x=torch.cat([graph.x,torch.zeros(graph.num_nodes,1)],1)
    model=auxiliary_model(task,model_id=task+'/fixture');model.args.update(configuration(severity,task))
    width=2 if severity=='scalar' else 1+model.args['outcome_decomposition_definition']['severity_distribution']['effective_atoms']
    model.forward=lambda graphs: graphs.x.new_zeros((graphs.num_nodes,2,width))
    validate_models(SimpleNamespace(decomposition=True),[task+'_success'],{task:model})
    configure_exports({task:model},True);configure_distribution_exports({task:model},True)
    failure,success=inference_gnn(match,model,device='cpu')
    write_exports(tmp_path,{task:model});write_distribution_exports(tmp_path,{task:model})
    rows=pd.read_parquet(tmp_path/'outcome_decomposition.parquet')
    for row in rows.itertuples():
        table=success if row.execution_branch else failure
        assert table.at[row.action_index,row.candidate_id]==pytest.approx(row.occurrence_probability*row.conditional_severity)
    class Axis:
        def __init__(self):self.calls=[]
        def annotate(self,*a,**kw):self.calls.append((a,kw))
    if severity=='distributional':
        dist_rows=pd.read_parquet(tmp_path/'outcome_distributions.parquet')
        support=model.args['outcome_decomposition_definition']['severity_distribution']
        assert set(dist_rows.support_definition_id)=={support['definition_sha256']}
        np.testing.assert_allclose([np.dot(p,support['atoms']) for p in dist_rows.probabilities],dist_rows.conditional_severity)
    axis=Axis();row=success.iloc[0];snapshot=match.tracking.iloc[[0]]
    annotate(axis,snapshot,'home',row)
    assert axis.calls and all(' x ' in a[0] for a,kw in axis.calls)
    assert all(kw['xytext']==(0,-3.5) for a,kw in axis.calls)


def test_annotation_scope_and_conflicts():
    from scripts.outcome_decomposition_visualization import validate_components,validate_options,validate_models
    with pytest.raises(ValueError,match='coach'):
        validate_options(SimpleNamespace(decomposition=True,coach_ratings=True))
    for name in ['outcome_success','outcome_failure','pass_score']:
        with pytest.raises(ValueError,match='individual'):
            validate_components(SimpleNamespace(decomposition=True),[name])
    with pytest.raises(ValueError,match='checkpoint'):
        validate_models(SimpleNamespace(decomposition=True),['outcome_scoring_success'],{})


@pytest.mark.parametrize('severity',['scalar','distributional'])
@pytest.mark.parametrize('task',['outcome_scoring','outcome_conceding'])
def test_real_cli_training(tmp_path,monkeypatch,severity,task):
    import runpy,sys
    import project_config as pc
    from test_all_task_training import test_fresh_cli_all_tasks_default_to_disk
    from training_state import load_checkpoint,CHECKPOINT_NAME
    original=runpy.run_path
    def run_decomposition(*args,**kwargs):
        sys.argv.extend(['--learning-mode','decomposition','--severity-mode',severity,'--use_xt','--return_type','next_10'])
        if severity=='distributional':sys.argv.extend(['--distribution-atoms','7'])
        for path in (tmp_path/'labels').rglob('*.pt'):
            labels=torch.load(path,weights_only=False)
            labels[:,LABEL_INDEX['success']]=torch.tensor([0.,1.])
            labels[:,LABEL_INDEX['scores_xt']]=torch.tensor([.001,.1])
            labels[:,LABEL_INDEX['concedes_xt']]=torch.tensor([.01,.2])
            torch.save(labels,path)
        monkeypatch.setattr(pc,'load_feature_run_metadata',lambda *a,**kw: {**metadata(),
            'stop_conditions':{'version':1,'available':{'next_10':['base']}}})
        return original(*args,**kwargs)
    monkeypatch.setattr(runpy,'run_path',run_decomposition)
    test_fresh_cli_all_tasks_default_to_disk(tmp_path,monkeypatch,task,'relationship-attention')
    checkpoint=load_checkpoint(tmp_path/'saved'/task/'verification'/CHECKPOINT_NAME)
    from models import utils
    directory=tmp_path/'saved'/task/'verification'
    monkeypatch.setattr(utils,'get_model_path',lambda *a:directory)
    loaded=utils.load_model(task+'/verification',device='cpu')
    assert loaded.args['outcome_decomposition_definition']==checkpoint['args']['outcome_decomposition_definition']
    for key,value in torch.load(directory/'best_weights.pt',weights_only=False).items():
        torch.testing.assert_close(loaded.state_dict()[key],value)
    assert checkpoint['args']['severity_mode']==severity
    assert checkpoint['metadata']['best_valid_metrics']['decomposition_loss']>=0
    assert checkpoint['metadata']['best_valid_metrics']['mean_soft_bce']==checkpoint['metadata']['best_valid_metrics']['ce_loss']


@pytest.mark.parametrize('severity',['scalar','distributional'])
def test_exact_resume(tmp_path,monkeypatch,severity):
    import runpy,sys,faulthandler
    import project_config as pc
    import models.utils as utils
    from test_all_task_training import test_fresh_cli_all_tasks_default_to_disk
    from training_state import load_checkpoint,CHECKPOINT_NAME
    original_run,real_epoch=runpy.run_path,utils.run_epoch
    def inject(*a,**kw):
        sys.argv.extend(['--learning-mode','decomposition','--severity-mode',severity,'--use_xt',
                         '--return_type','next_10','--n_epochs','2'])
        if severity=='distributional':sys.argv.extend(['--distribution-atoms','7'])
        label_root=Path(sys.argv[sys.argv.index('--label_dir')+1])
        for path in label_root.glob('*.pt'):
            labels=torch.load(path,weights_only=False)
            labels[:,LABEL_INDEX['success']]=torch.tensor([0.,1.])
            labels[:,LABEL_INDEX['concedes_xt']]=torch.tensor([.001,.1])
            torch.save(labels,path)
        monkeypatch.setattr(pc,'load_feature_run_metadata',lambda *a,**kw: {**metadata(),
            'stop_conditions':{'version':1,'available':{'next_10':['base']}}})
        return original_run(*a,**kw)
    monkeypatch.setattr(runpy,'run_path',inject)
    baseline_root,resumed_root=tmp_path/'baseline',tmp_path/'resumed'
    baseline_root.mkdir();resumed_root.mkdir()
    task='outcome_conceding'
    test_fresh_cli_all_tasks_default_to_disk(baseline_root,monkeypatch,task,'gat')
    baseline=load_checkpoint(baseline_root/'saved'/task/'verification'/CHECKPOINT_NAME)
    def interrupt(args,*a,**kw):
        if args._epoch==2:raise RuntimeError('decomposition interruption')
        return real_epoch(args,*a,**kw)
    monkeypatch.setattr(utils,'run_epoch',interrupt)
    with pytest.raises(RuntimeError,match='decomposition interruption'):
        test_fresh_cli_all_tasks_default_to_disk(resumed_root,monkeypatch,task,'gat')
    faulthandler.disable()
    checkpoint_path=resumed_root/'saved'/task/'verification'/CHECKPOINT_NAME
    interrupted=load_checkpoint(checkpoint_path)
    monkeypatch.setattr(utils,'run_epoch',real_epoch)
    monkeypatch.setattr(sys,'argv',['train.py','--resume-checkpoint',str(checkpoint_path)])
    try:original_run(str(Path(__file__).resolve().parents[1]/'train.py'),run_name='__main__')
    finally:faulthandler.disable()
    resumed=load_checkpoint(checkpoint_path)
    assert resumed['finished']
    assert resumed['args']['outcome_decomposition_definition']==interrupted['args']['outcome_decomposition_definition']
    for key,value in baseline['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
    for key,value in baseline['optimizer']['state'].items():
        for name,tensor in value.items():
            torch.testing.assert_close(tensor,resumed['optimizer']['state'][key][name],rtol=0,atol=0)


from test_training_refit import sources


@pytest.mark.parametrize('severity',['scalar','distributional'])
def test_source_refit_inherits_definition(sources,severity):
    from training_refit import prepare_refit
    from models.outcome_learning import distribution_identity
    create,_=sources
    directory,args,record=create('outcome_scoring/source')
    args.update(configuration(severity,'outcome_scoring'))
    record.update(distribution_identity(args))
    (directory/'args.json').write_text(json.dumps(args))
    (directory/'metadata.json').write_text(json.dumps(record))
    refit=prepare_refit('outcome_scoring/source','refit')
    assert refit['args']['outcome_decomposition_definition']==args['outcome_decomposition_definition']


def test_saved_annotation_attachment_and_mismatch():
    import pandas as pd
    from scripts.outcome_decomposition_visualization import attach_saved
    table=pd.DataFrame([[.1]],index=[7],columns=['home_1'])
    rows=[dict(scene_id='scene',task='outcome_conceding',execution_branch=1,frame_id=7,
               candidate_id='home_1',prediction=.1,occurrence_probability=.5,conditional_severity=.2)]
    attach_saved({'outcome_conceding_success':table},rows,'scene')
    assert table.attrs['outcome_decomposition'][7]['home_1']==[.5,.2]
    mixed={'outcome_conceding_success':table,'outcome_scoring_success':table.copy()}
    attach_saved(mixed,rows,'scene',components=['outcome_conceding_success'])
    with pytest.raises(ValueError,match='Ambiguous'):
        attach_saved({'outcome_conceding_success':table},rows+rows,'scene')
    rows[0]['prediction']=.2
    with pytest.raises(ValueError,match='disagree'):
        attach_saved({'outcome_conceding_success':table},rows,'scene')
