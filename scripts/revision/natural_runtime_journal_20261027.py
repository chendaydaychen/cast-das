"""Owner-finalized actual API/value/version journal for frozen natural replay."""
import hashlib
import json
import threading
import time


def value_hash(value, exists=True):
    return hashlib.sha256(json.dumps([bool(exists), str(value)], separators=(',', ':')).encode()).hexdigest()


class NaturalRuntimeJournal:
    """Mixin BEFORE the selected CC manager; no CC/protection decisions here."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.natural_initial = {}; self.natural_attempts = []
        self.natural_local = threading.local(); self.natural_lock = threading.RLock()

    def register_object(self, key, value, **kwargs):
        result = super().register_object(key, value, **kwargs)
        stored = self.store.get(str(key))
        self.natural_initial.setdefault(str(key), {'value': str(stored.value),
            'version': int(stored.version), 'exists': bool(stored.exists)})
        return result

    def begin(self, *args, **kwargs):
        started = time.monotonic_ns(); txn = super().begin(*args, **kwargs)
        row = {'task_id': txn.task_id, 'tid': txn.context.tid, 'attempt': txn.context.retry_count,
            'sequence': getattr(self.natural_local, 'sequence', None), 'start_ns': started,
            'owner_ident': threading.get_ident(), 'worker_tid': threading.get_native_id(),
            'affinity_start': sorted(__import__('os').sched_getaffinity(0)), 'ops': [], 'waits': []}
        txn.metadata['_natural_journal_row'] = row
        with self.natural_lock: self.natural_attempts.append((txn, row))
        original_read, original_write, original_abort = txn.read, txn.write, txn.abort
        def access(kind, key, value=None, **options):
            begin = time.monotonic_ns(); result_value = None; version = None; returned = False; exists = None
            try:
                result = original_read(key, **options) if kind == 'read' else original_write(key, value, **options)
                result_value = str(result.value) if kind == 'read' else str(value)
                if kind == 'read': version = int(result.version); exists = bool(result.exists)
                returned = True; return result
            finally:
                operation = {'kind': kind, 'key': str(key), 'value': result_value,
                    'version': version, 'exists': exists, 'returned': returned,
                    'start_ns': begin, 'end_ns': time.monotonic_ns()}
                if kind == 'write' and str(key) in txn.metadata.get('runtime_write_identities', {}):
                    operation['logical_write_identity'] = txn.metadata['runtime_write_identities'][str(key)]
                row['ops'].append(operation)
        def abort(*args, **kwargs):
            try: return original_abort(*args, **kwargs)
            finally: self._natural_finish(txn)
        txn.read = lambda key, **options: access('read', key, **options)
        txn.write = lambda key, value, **options: access('write', key, value, **options)
        txn.abort = abort
        return txn

    def _natural_finish(self, txn):
        row = txn.metadata.get('_natural_journal_row')
        if row is not None and txn.result is not None and not txn.metadata.get('_defer_commit_timing_finalize', False):
            if threading.get_ident() == row['owner_ident']:
                row['end_ns'] = time.monotonic_ns()
                row['affinity_end'] = sorted(__import__('os').sched_getaffinity(0))

    def _finalize_commit_timing(self, txn):
        try: return super()._finalize_commit_timing(txn)
        finally: self._natural_finish(txn)

    def commit(self, txn, **kwargs):
        try: return super().commit(txn, **kwargs)
        finally: self._natural_finish(txn)

    def export_natural_journal(self):
        rows = []
        for txn, source in self.natural_attempts:
            if txn.result is None or 'end_ns' not in source:
                raise ValueError('Unfinished/unfinalized natural attempt')
            versions = txn.metadata.get('runtime_published_versions')
            if versions is None:
                versions = {key: int(write.base_version)+1 for key, write in txn.write_set.items()}
            rows.append(dict(source, committed=bool(txn.result.committed), reason=str(txn.result.reason),
                actual_strategy=txn.result.strategy, published_versions=dict(versions) if txn.result.committed else {}))
            if txn.result.committed and 'runtime_published_identities' in txn.metadata:
                rows[-1]['published_logical_identities'] = dict(txn.metadata['runtime_published_identities'])
        if len({row['tid'] for row in rows}) != len(rows): raise ValueError('Natural attempt identity reused')
        final = {key: {'value': str(value.value), 'version': int(value.version), 'exists': bool(value.exists)}
            for key in self._catalog for value in [self.store.get(key)]}
        return {'initial': self.natural_initial, 'attempts': rows, 'final': final}


def actual_version_history(journal):
    """Exclude self reads; preserve actual external returns, not snapshot guesses."""
    attempts = []
    for row in journal['attempts']:
        reads = {}; writes = {}
        for op in row['ops']:
            if not op['returned']: continue
            key = op['key']
            if op['kind'] == 'write': writes[key] = op['value']
            elif key not in writes:
                read = {'version': op['version'], 'value_hash': value_hash(op['value'], op['exists'])}
                if key in reads and reads[key] != read: raise ValueError('Nonrepeatable external point read')
                reads[key] = read
        if row['committed'] and set(writes) != set(row['published_versions']):
            raise ValueError('Missing actual publication versions')
        attempts.append({'id': row['tid'], 'committed': row['committed'],
            'start_ns': row['start_ns'], 'end_ns': row['end_ns'], 'reads': reads,
            'writes': {key: {'version': row['published_versions'].get(key, 0),
                'value_hash': value_hash(value)} for key, value in writes.items()}})
    def state(rows):
        return {key: {'version': row['version'], 'value_hash': value_hash(row['value'], row['exists'])}
            for key, row in rows.items()}
    return {'initial': state(journal['initial']), 'attempts': attempts, 'final': state(journal['final'])}
