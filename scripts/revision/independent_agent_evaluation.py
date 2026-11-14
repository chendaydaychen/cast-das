"""Validate an explicitly frozen, company-disjoint live evaluation capture."""
import hashlib
import json

from agent.workloads.frozen_credit_agent import is_complete_aborted_intent


def load_independent_capture(path, *, expected_sha256, manifest_path, training_folder,
                             capture_folder, allow_complete_aborted_capture=False):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('independent capture SHA256 differs from declared input')
    manifest = json.loads(manifest_path.read_text())
    required = {'protocol.json', 'training_report.json', 'training_transitions.json',
                'full_policy.json', 'nophase_policy.json', 'linear_policy.json'}
    if (manifest.get('model_requests_started') is not False
            or not required.issubset(manifest.get('sha256', {}))):
        raise ValueError('pre-capture frozen manifest required')
    for name in required:
        if hashlib.sha256((training_folder / name).read_bytes()).hexdigest() != manifest['sha256'][name]:
            raise ValueError('learned artifact changed after capture manifest')
    source = json.loads(path.read_text())
    train = json.loads((capture_folder / 'train_live.json').read_text())
    reference = json.loads((capture_folder / 'evaluation_live.json').read_text())
    if (source.get('mode') != 'live' or not source.get('agent_directed')
            or source.get('config') != reference.get('config')
            or source.get('model') != reference.get('model')
            or source.get('multi_storage') != reference.get('multi_storage')
            or not source.get('tasks')):
        raise ValueError('independent model/config/storage capture contract differs')
    training_companies = {row['task']['company_id'] for row in train['tasks']}
    training_protocol = json.loads((training_folder/'protocol.json').read_text())
    training_companies.update(training_protocol.get('additional_training_company_ids', []))
    companies = {row['task']['company_id'] for row in source['tasks']}
    if training_companies & companies:
        raise ValueError('independent evaluation overlaps training companies')
    if (len(source['tasks']) != manifest['tasks'] or source.get('workers') != manifest['workers']
            or (manifest.get('company_id') is not None and companies != {manifest['company_id']})):
        raise ValueError('independent capture differs from predeclared task/company plan')
    for row in source['tasks']:
        if not row.get('committed') and not (allow_complete_aborted_capture and is_complete_aborted_intent(row)):
            raise ValueError('complete agent intent stream required; failed planner cannot be replayed')
    return source, dict(capture_sha256=actual,
        pre_capture_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        training_company_ids=sorted(training_companies), evaluation_company_ids=sorted(companies),
        recorded_commits=sum(row.get('committed') is True for row in source['tasks']),
        recorded_complete_aborted_intents=sum(row.get('committed') is False for row in source['tasks']),
        allow_complete_aborted_capture=allow_complete_aborted_capture,
        scope='new model-selected intents after frozen-model manifest; frozen replay, not online replanning')
