# Aegis

Aegis is a compact research prototype for agent-side transaction semantics  
and concurrency control over a versioned KV backend. The storage layer is kept  
small on purpose: it exposes object reads, version reads, and conditional batch  
writes. Transaction boundaries, read/write sets, conflict validation, retry  
logic, and concurrency-control selection stay in the Python agent runtime.

### Storage backends

The default backend is an embedded DBx1000 versioned KV store. With multiple
DBx1000 instances, each object is registered with a `storage_id` and the native
coordinator validates and publishes batches across the participating instances.
This path provides in-process atomicity for the in-memory stores.

The opt-in durable router runs on Linux and uses SQLite for business records
and an atomically replaced, fsync-backed JSON KV file for audit records. A local
coordinator records the commit decision before installing either participant's
updates. On restart it discards undecided batches and replays decided batches
before serving reads. The coordinator requires exclusive ownership of the
participant files.

### CreditCrew multi-storage replay

Build the native extension with `bash build.sh`. Supply a completed,
model-directed CreditCrew capture recorded with `--multi-storage` and a directory
containing the frozen `full_int16_table.json` policy. The same capture and policy
can be replayed through either backend without calling an LLM. Credit and
business objects use the default participant. Audit and compliance objects use
the audit participant.

```bash
python3 scripts/revision/run_current_credit_replay_20261110.py \
  --capture /path/to/capture.json --model-dir /path/to/policies \
  --arms full silo --workers 2 --max-retries 5 \
  --storage native --audit-storage --output /tmp/credit-native

python3 scripts/revision/run_current_credit_replay_20261110.py \
  --capture /path/to/capture.json --model-dir /path/to/policies \
  --arms full silo --workers 2 --max-retries 5 \
  --storage durable --output /tmp/credit-durable
```

Both commands check that the recorded history is serializable and that at least
one committed transaction writes to both participants. The durable run also
reopens the store and verifies persisted values and versions. Wall-clock replay
measurements include the captured reasoning and tool waits. Capture files,
policy artifacts, and generated outputs are supplied separately.

