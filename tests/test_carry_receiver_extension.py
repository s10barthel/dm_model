from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from datatools import graph_feature as gf
from datatools.carry_label_extension import load_carry_reference
from datatools.graph_validation import validation_stats, GraphValidationError
from datatools.match import Match
from datatools.receiver_reuse import PROVENANCE, prediction_provenance, reusable_predictions
from datatools.trajectory import TRAJECTORY_SPEED
from scripts import generate_relevant_features as generator
import test_feature_run_extension as extension_tests
from test_feature_run_extension import make_args, make_metadata


def actions():
    return pd.DataFrame(dict(action_type=['pass','pass','dribble'], success=[False,False,True],
        original_event_id=['100.0','101','carry:1'], frame_id=[10,20,30], period_id=[1,1,1],
        object_id=['home_1']*3, observed_end_frame_id=[15.,25.,35.], trajectory_end_frame_id=[15.,25.,35.],
        pass_endpoint_definition=['official','official',''], pass_duration=[.2,.2,.2],
        intent_id=['home_2',None,'home_1'], model_receiver_reason=[None,'missing_launch_frame',None],
        spadl_type=['pass','pass','ball_carry'], carry_definition=[None,None,'test-carries']), index=[1,2,10])


@pytest.fixture
def carry_root(tmp_path):
    metadata=dict(carry_variant=dict(available=True,definition='test-carries'),
        intended_receiver_modes=['original','angle_only'], return_types=['next_10'],
        graph_schema=dict(node_in_dim=26,edge_in_dim=2,add_v_edge_features=False,
                          edge_feature_names=['distance','is_teammate']))
    (tmp_path/'metadata.json').write_text(json.dumps(metadata))
    table=actions()
    labels=torch.zeros((3,31)); labels[:,0]=torch.tensor([1,2,10])
    path=gf.get_resolved_action_path('match',intended_receiver_mode='original',root=tmp_path,use_carries=True)
    path.parent.mkdir(); table.to_parquet(path)
    folder=gf.get_action_label_dir('next_10',intended_receiver_mode='original',root=tmp_path,use_carries=True)
    folder.mkdir(); torch.save(labels,folder/'match.pt')
    edges=torch.cartesian_prod(torch.arange(2),torch.arange(2)).t().contiguous()
    graph=Data(x=torch.zeros((2,26)),edge_index=edges,edge_attr=torch.ones((4,2)))
    graph.edge_feature_names=['distance','is_teammate']; graph.node_ids=['home_1','home_2']
    for folder in (gf.get_action_graph_dir(tmp_path,use_carries=True), gf.get_post_action_graph_dir(tmp_path,use_carries=True)):
        folder.mkdir(); torch.save([graph,None,graph],folder/'match.pt')
    return tmp_path,table,labels,metadata


def test_carry_save_preserves_graphs_and_ordinary_outputs(carry_root):
    root,table,labels,metadata=carry_root
    original={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.pt')}
    match=SimpleNamespace(carry_reference_labels=labels,carry_reference_metadata=metadata,
                          graph_validation_stats=validation_stats())
    with patch.object(gf,'construct_graph_features',side_effect=AssertionError('graph reconstruction')):
        gf.save_labels_only_artifacts(match,{'model':table},{('model','next_10'):labels},
            ['model'],['next_10'],'match',root,use_carries=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in original.items())
    assert not gf.get_action_label_dir('next_10',intended_receiver_mode='model',root=root).exists()
    assert gf.get_resolved_action_path('match',intended_receiver_mode='model',root=root,use_carries=True).exists()
    assert match.graph_validation_stats['checked_files']==2
    assert match.graph_validation_stats['verified_output_files']==0


def test_reference_requires_order_coverage_and_existing_graphs(carry_root):
    root,table,labels,_=carry_root
    loaded,reference,_=load_carry_reference(root,'match')
    assert loaded.index.tolist()==[1,2,10] and torch.equal(reference,labels)
    path=gf.get_action_label_dir('next_10',intended_receiver_mode='original',root=root,use_carries=True)/'match.pt'
    torch.save(labels.flip(0),path)
    with pytest.raises(ValueError,match='ordering'): load_carry_reference(root,'match')
    torch.save(labels,path)
    (gf.get_post_action_graph_dir(root,use_carries=True)/'match.pt').unlink()
    with pytest.raises(FileNotFoundError,match='retained carry graph'): load_carry_reference(root,'match')


def test_bad_graph_or_order_publishes_no_carry_labels(carry_root):
    root,table,labels,metadata=carry_root
    match=SimpleNamespace(carry_reference_labels=labels,carry_reference_metadata=metadata)
    with pytest.raises(ValueError,match='ordering|sorted'):
        gf.save_labels_only_artifacts(match,{'model':table},{('model','next_10'):labels.flip(0)},
            ['model'],['next_10'],'match',root,use_carries=True)
    torch.save([],gf.get_post_action_graph_dir(root,use_carries=True)/'match.pt')
    with pytest.raises(GraphValidationError,match='row count'):
        gf.save_labels_only_artifacts(match,{'model':table},{('model','next_10'):labels},
            ['model'],['next_10'],'match',root,use_carries=True)
    assert not gf.get_resolved_action_path('match',intended_receiver_mode='model',root=root,use_carries=True).exists()


def test_carry_enabled_match_without_synthetic_carries_is_supported(carry_root):
    root,table,_,_=carry_root
    table['spadl_type']='pass'; table=table.drop(columns='carry_definition')
    table.to_parquet(gf.get_resolved_action_path('match',intended_receiver_mode='original',root=root,use_carries=True))
    loaded,_,_=load_carry_reference(root,'match')
    assert loaded.index.tolist()==[1,2,10]


def test_provenance_roundtrip_and_input_invalidation(carry_root):
    root,table,_,_=carry_root
    provenance=prediction_provenance(root,'match','success_intent/test','attempt')
    table.attrs[PROVENANCE]=provenance
    table.to_parquet(root/'predictions.parquet')
    source=pd.read_parquet(root/'predictions.parquet')
    assert reusable_predictions(table.iloc[::-1],source,provenance)=={2:(None,'missing_launch_frame'),1:('home_2',None)}
    assert reusable_predictions(table,source,dict(provenance,model_id='other'))=={}
    assert reusable_predictions(table,source,dict(provenance,attempt_id='other'))=={}
    assert reusable_predictions(table,source,None)=={}
    metadata=json.loads((root/'metadata.json').read_text()); metadata['reception_policy']='changed'
    (root/'metadata.json').write_text(json.dumps(metadata))
    assert reusable_predictions(table,source,prediction_provenance(root,'match','success_intent/test','attempt'))=={}
    changed=table.copy(); changed.loc[1,'frame_id']=11
    assert reusable_predictions(changed,source,provenance)=={2:(None,'missing_launch_frame')}
    source=source.drop(2)
    assert reusable_predictions(table,source,provenance)=={1:('home_2',None)}


def test_duplicate_failed_pass_identities_are_rejected():
    table=actions(); table.loc[2,'original_event_id']='100'
    with pytest.raises(ValueError,match='Duplicate'): reusable_predictions(table,None,None)


def test_fully_reused_decisions_skip_model_loading_and_forward():
    table=actions(); match=SimpleNamespace(_get_cached_intended_receiver_model=Mock(side_effect=AssertionError('loaded model')))
    result=Match._apply_intended_receiver_model(match,table,'success_intent/test',
        precomputed_predictions={1:('home_2',None),2:(None,'missing_launch_frame')})
    assert result.loc[1,'intent_id']=='home_2' and result.loc[10,'intent_id']=='home_1'
    stats=result.attrs['intended_receiver_stats']
    assert stats['model_predictions_reused']==2 and stats['model_predictions_inferred']==0
    assert stats['model_unavailable_predictions']==1


def test_partial_reuse_constructs_only_missing_model_inputs():
    table=actions(); model=SimpleNamespace(args={'trajectory_speed_definition':TRAJECTORY_SPEED})
    match=SimpleNamespace(actions=table,_get_cached_intended_receiver_model=Mock(return_value=(
        model,dict(edge_in_dim=2,add_v_edge_features=False))))
    with patch.object(gf,'construct_graph_for_action',return_value=None) as construct:
        result=Match._apply_intended_receiver_model(match,table,'success_intent/test',precomputed_predictions={1:('home_2',None)})
    assert [call.args[1] for call in construct.call_args_list]==[2]
    assert result.attrs['intended_receiver_stats']['model_predictions_inferred']==1


@pytest.mark.parametrize('replace',[False,True])
@pytest.mark.parametrize('in_place',[False,True])
def test_carry_receiver_stages_are_planned_upfront(replace,in_place):
    metadata=make_metadata(['next_10'],['original','angle_only','model'] if replace else None,
                           'success_intent/old' if replace else None)
    metadata['carry_variant']=dict(available=True)
    args=make_args(model_id='success_intent/new',replace_model=replace,in_place=in_place,
                   run_id=None if in_place else 'derived')
    plan=extension_tests.FeatureRunExtensionPlanTests().build_plan(args,metadata)
    assert len(plan.command_steps)==5 and len(plan.commands)==5
    for step in plan.command_steps[-2:]:
        assert '--labels-only' in step.command and '--use-carries' in step.command
        assert '--post_action' not in step.command
        assert step.command[step.command.index('--receiver-extension-attempt')+1]==plan.receiver_extension_attempt


def test_extension_reports_whole_workload_once(capsys):
    steps=[generator.FeatureGenerationStep(name,['python','feature.py']) for name in ('ordinary','carry')]
    with patch.object(generator,'run_command'):
        generator.run_generation_steps(steps)
    output=capsys.readouterr().out
    assert output.index('2/2: carry')<output.index('Feature generation step 1/2')
    assert 'Finished step 2/2' in output and 'total elapsed' in output


def test_receiver_extension_preserves_source_height_threshold():
    metadata=make_metadata(); metadata['pass_height_threshold_meters']=1.7
    metadata['carry_variant']=dict(available=True)
    args=make_args(model_id='success_intent/new')
    plan=extension_tests.FeatureRunExtensionPlanTests().build_plan(args,metadata)
    assert generator.extension_pass_height_threshold(args,plan)==1.7
    assert all(step.command[step.command.index('--pass-height-threshold')+1]=='1.7' for step in plan.command_steps)


def test_audit_rejects_missing_or_partial_stage_reports(tmp_path,monkeypatch):
    plan=SimpleNamespace(target_run_id='target',base_metadata={},base_intended_receiver_modes=['original'],
        receiver_extension_attempt='attempt',base_return_types=['next_10'],final_return_types=['next_10'],
        intended_receiver_model_id='success_intent/test')
    monkeypatch.setattr(generator,'get_feature_run_root',lambda _:tmp_path)
    monkeypatch.setattr(generator,'resolve_split_manifest',lambda **_:dict(train=['match'],test=[]))
    with pytest.raises(FileNotFoundError): generator.audit_receiver_extension(plan)
    reports=tmp_path/'generation_reports'; reports.mkdir()
    (reports/'train_base.json').write_text(json.dumps(dict(receiver_extension_attempt='attempt',
        expected_matches=['match'],successful_matches=0,skipped_matches=[dict(match_id='match')])) )
    with pytest.raises(ValueError,match='coverage'): generator.audit_receiver_extension(plan)


def test_audit_checks_generated_label_order_and_provenance(tmp_path,monkeypatch):
    plan=SimpleNamespace(target_run_id='target',base_metadata={},base_intended_receiver_modes=['original'],
        receiver_extension_attempt='attempt',base_return_types=['next_10'],final_return_types=['next_10'],
        intended_receiver_model_id='success_intent/test')
    monkeypatch.setattr(generator,'get_feature_run_root',lambda _:tmp_path)
    monkeypatch.setattr(generator,'resolve_split_manifest',lambda **_:dict(train=['match'],test=[]))
    reports=tmp_path/'generation_reports'; reports.mkdir()
    for split,variant,ids in [('train','base',['match']),('train','intent_train_augmented',['match']),('test','base',[])]:
        (reports/f'{split}_{variant}.json').write_text(json.dumps(dict(receiver_extension_attempt='attempt',
            expected_matches=ids,successful_matches=len(ids),skipped_matches=[])))
    labels=torch.zeros((3,31)); labels[:,0]=torch.tensor([1,2,10])
    for builder in (gf.get_action_label_dir,gf.get_intent_train_label_dir):
        for mode in ('original','model'):
            folder=builder('next_10',intended_receiver_mode=mode,root=tmp_path); folder.mkdir()
            torch.save(labels,folder/'match.pt')
    path=gf.get_resolved_action_path('match',intended_receiver_mode='model',root=tmp_path)
    path.parent.mkdir(); table=actions(); table.attrs[PROVENANCE]=dict(attempt_id='attempt',model_id='success_intent/test')
    table.to_parquet(path)
    generator.audit_receiver_extension(plan)
    folder=gf.get_intent_train_label_dir('next_10',intended_receiver_mode='model',root=tmp_path)
    torch.save(labels.flip(0),folder/'match.pt')
    with pytest.raises(ValueError,match='alignment'): generator.audit_receiver_extension(plan)


def test_failed_completion_audit_does_not_publish_latest(tmp_path,monkeypatch):
    args=make_args(model_id='success_intent/test')
    metadata=make_metadata()
    source=tmp_path/'base'; source.mkdir(); (source/'metadata.json').write_text(json.dumps(metadata))
    monkeypatch.setattr(generator,'resolve_feature_run_id',lambda *a,**kw:'base')
    monkeypatch.setattr(generator,'load_feature_run_metadata',lambda *a,**kw:metadata)
    monkeypatch.setattr(generator,'get_feature_run_root',lambda run:tmp_path/run)
    monkeypatch.setattr(generator,'run_command',lambda _:None)
    monkeypatch.setattr(generator,'audit_receiver_extension',Mock(side_effect=ValueError('coverage audit failed')))
    latest=Mock(); monkeypatch.setattr(generator,'write_latest_run',latest)
    with pytest.raises(ValueError,match='coverage audit'): generator.run_extension_generation(args)
    result=json.loads((tmp_path/'derived/metadata.json').read_text())
    assert result['status']=='failed' and 'model' not in result['intended_receiver_modes']
    latest.assert_not_called()
