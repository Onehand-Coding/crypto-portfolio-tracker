import { useRef, useState } from 'react';
import { Panel } from '../components/Panel';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import { formatSigned, formatUsd, NULL_GLYPH} from '../lib/format';
import type {
  CleanupResponse, CleanupStatsResponse, ImportResponse, SnapshotDeleteResponse,
  SnapshotRow, SnapshotsResponse,
} from '../types';

const control = {
  background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
  borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
  padding: 'var(--space-2) var(--space-3)', fontSize: '13px',
} as const;

/** Persisted-snapshot outcome. Local: the shared types file is out of scope. */
interface SnapshotSaveResponse {
  saved: boolean;
  timestamp: string | null;
  error: string | null;
}

/** Import transactions or holdings from a CSV/Excel file. */
function ImportPanel({ onDone }: { onDone: () => void }) {
  const [kind, setKind] = useState('transactions');
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function submit() {
    const file = fileRef.current?.files?.[0];
    if (!file) { setMessage('Choose a file first.'); return; }
    setImporting(true);
    setMessage(null);
    try {
      const body = new FormData();
      body.append('file', file);
      const res = await fetch(`/api/system/import/${kind}`, { method: 'POST', body });
      const data = (await res.json()) as ImportResponse;
      setMessage(data.success
        ? `Imported ${data.rows_affected} ${kind} row${data.rows_affected === 1 ? '' : 's'}.`
        : `Import failed: ${data.error ?? res.statusText}`);
      if (data.success) { onDone(); if (fileRef.current) fileRef.current.value = ''; }
    } catch (e) {
      setMessage(`Import failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setImporting(false);
    }
  }

  return (
    <Panel title="Import data">
      <p className="font-ui text-sm"
         style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
        Load a CSV or Excel file exported from this app. A backup is taken before
        anything is written, so an import can be undone from the System screen.
      </p>
      <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
        <select value={kind} onChange={(e) => setKind(e.target.value)}
                className="font-mono" style={{ ...control, minWidth: '150px' }}>
          <option value="transactions">Transactions</option>
          <option value="holdings">Holdings</option>
        </select>
        <input ref={fileRef} type="file" accept=".csv,.xlsx,.xls"
               className="font-ui text-sm" style={{ color: 'var(--text-secondary)' }} />
        <Button onClick={submit} disabled={importing}>
          {importing ? 'Importing…' : 'Import'}
        </Button>
        {message && (
          <span className="font-ui" style={{ fontSize: '13px',
                   color: message.startsWith('Import failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
            {message}
          </span>
        )}
      </div>
    </Panel>
  );
}

/** Build a CSV from header + rows and trigger a download, no server round-trip. */
function downloadRows(filename: string, headers: string[], rows: (string | number | null)[][]) {
  const esc = (v: string | number | null) => {
    const s = v === null || v === undefined ? '' : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [headers.join(','), ...rows.map((r) => r.map(esc).join(','))];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Export the currently listed snapshots as a local CSV file. */
function exportSnapshots(rows: SnapshotRow[]) {
  downloadRows(
    `snapshots_${new Date().toISOString().slice(0, 10)}.csv`,
    ['timestamp', 'total_value_usd', 'total_cost_basis_usd',
     'unrealized_pl_usd', 'unrealized_pl_percent'],
    rows.map((r) => [r.timestamp, r.total_value_usd, r.total_cost_basis_usd,
                     r.unrealized_pl_usd, r.unrealized_pl_percent]),
  );
}

/** Portfolio snapshots with per-row delete. */
function SnapshotsPanel() {
  const { data, reload } = useApi<SnapshotsResponse>('/api/system/snapshots');
  const [confirm, setConfirm] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function save() {
    setSaving(true);
    setMessage(null);
    try {
      const res = await apiPost<SnapshotSaveResponse>('/api/system/snapshot/save');
      setMessage(res.saved
        ? `Snapshot saved${res.timestamp ? ` at ${res.timestamp.slice(0, 19).replace('T', ' ')}` : ''}.`
        : `Save failed${res.error ? `: ${res.error}` : '.'}`);
      if (res.saved) reload();
    } catch (e) {
      setMessage(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  }

  async function del(row: SnapshotRow) {
    setBusy(true);
    setMessage(null);
    try {
      const res = await apiPost<SnapshotDeleteResponse>('/api/system/snapshots/delete', {
        confirm: true, ...row,
      });
      setMessage(res.deleted > 0 ? `Deleted ${res.deleted} snapshot.`
                                 : `Nothing deleted${res.error ? `: ${res.error}` : '.'}`);
      setConfirm(null);
      reload();
    } catch (e) {
      setMessage(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title={`Snapshots (${data?.count ?? 0})`}>
      {message && (
        <p className="font-ui" style={{ fontSize: '13px', margin: '0 0 var(--space-3) 0',
                 color: (message.startsWith('Delete failed') || message.startsWith('Save failed'))
                   ? 'var(--negative)' : 'var(--text-secondary)' }}>
          {message}
        </p>
      )}
      <div className="flex items-center justify-end"
           style={{ gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
        <Button onClick={save} disabled={saving}>
          {saving ? 'Saving…' : 'Save snapshot'}
        </Button>
        {data && data.rows.length > 0 && (
          <Button variant="secondary" onClick={() => exportSnapshots(data.rows)}>
            Export CSV
          </Button>
        )}
      </div>
      {!data ? <Empty>Loading…</Empty> : data.rows.length === 0 ? (
        <Empty>No snapshots recorded yet.</Empty>
      ) : (
        <>
          <div className="table-scroll" style={{ maxHeight: '420px', overflowY: 'auto' }}>
          <table className="data">
            <thead>
              <tr>
                <th className="text-left">Timestamp</th>
                <th className="text-right">Value</th>
                <th className="text-right">Unrealized</th>
                <th className="text-right">Delete</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row, i) => {
                const key = `${row.timestamp}-${i}`;
                return (
                  <tr key={key}>
                    <td className="text-left" style={{ color: 'var(--text-secondary)' }}>
                      {row.timestamp ? row.timestamp.slice(0, 19).replace('T', ' ') : NULL_GLYPH}
                    </td>
                    <td className="text-right">{formatUsd(row.total_value_usd)}</td>
                    <td className="text-right"
                        style={{ color: (row.unrealized_pl_usd ?? 0) >= 0 ? 'var(--positive)' : 'var(--negative)' }}>
                      {formatSigned(row.unrealized_pl_usd)}
                    </td>
                    <td className="text-right">
                      {confirm === key ? (
                        <span className="flex items-center justify-end" style={{ gap: 'var(--space-2)' }}>
                          <button onClick={() => del(row)} disabled={busy}
                            className="font-ui" style={{ background: 'color-mix(in srgb, var(--negative) 18%, transparent)',
                              color: 'var(--negative)', border: '1px solid color-mix(in srgb, var(--negative) 35%, transparent)',
                              borderRadius: 'var(--radius-control)', padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}>
                            {busy ? '…' : 'Confirm'}
                          </button>
                          <button onClick={() => setConfirm(null)}
                            className="font-ui" style={{ background: 'transparent', color: 'var(--text-tertiary)',
                              border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                              padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}>
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button onClick={() => { setConfirm(key); setMessage(null); }}
                          className="font-ui" style={{ background: 'transparent', color: 'var(--text-secondary)',
                            border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-control)',
                            padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}>
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </>
      )}
    </Panel>
  );
}

/** Retention-based data cleanup, behind a confirm. */
function CleanupPanel() {
  const { data, reload } = useApi<CleanupStatsResponse>('/api/system/cleanup');
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setMessage(null);
    try {
      const res = await apiPost<CleanupResponse>('/api/system/cleanup', { confirm: true });
      setMessage(res.success ? (res.message ?? 'Cleanup complete.')
                             : `Cleanup failed: ${res.error ?? 'unknown error'}`);
      setConfirm(false);
      reload();
    } catch (e) {
      setMessage(`Cleanup failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  const entries = Object.entries(data?.stats ?? {});
  return (
    <Panel title="Data cleanup">
      <p className="font-ui text-sm" style={{ color: 'var(--text-secondary)', margin: '0 0 var(--space-3) 0' }}>
        {data
          ? data.enabled
            ? `Removes data older than the ${data.cleanup_days}-day retention. The database is backed up first.`
            : 'Cleanup is disabled (retention is 0). Set a retention period in Settings to enable it.'
          : 'Loading…'}
      </p>
      {entries.length > 0 && (
        <div className="flex flex-wrap" style={{ gap: 'var(--space-5)', marginBottom: 'var(--space-4)' }}>
          {entries.map(([k, v]) => (
            <div key={k} className="flex flex-col" style={{ gap: '2px' }}>
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '10px',
                                                 letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                {k.replace(/_/g, ' ')}
              </span>
              <span className="font-mono" style={{ fontSize: '14px' }}>{String(v)}</span>
            </div>
          ))}
        </div>
      )}
      {message && (
        <p className="font-ui" style={{ fontSize: '13px', margin: '0 0 var(--space-3) 0',
                 color: message.startsWith('Cleanup failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
          {message}
        </p>
      )}
      {confirm ? (
        <div className="flex items-center" style={{ gap: 'var(--space-3)' }}>
          <span className="font-ui text-sm" style={{ color: 'var(--warning)' }}>
            Delete data older than retention?
          </span>
          <Button onClick={run} disabled={busy}>{busy ? 'Running…' : 'Confirm cleanup'}</Button>
          <Button variant="secondary" onClick={() => setConfirm(false)}>Cancel</Button>
        </div>
      ) : (
        <Button variant="secondary" onClick={() => { setConfirm(true); setMessage(null); }}
                disabled={!data?.enabled}>
          Run cleanup
        </Button>
      )}
    </Panel>
  );
}

export function DataManage() {
  const { error, reload } = useApi<SnapshotsResponse>('/api/system/snapshots');
  if (error) return <ErrorPanel title="Manage data" message={`Failed to load: ${error}`} />;

  return (
    <>
      <ScreenHeader title="Manage data"
                    subtitle="Import, snapshots and retention cleanup" />
      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <ImportPanel onDone={reload} />
        <SnapshotsPanel />
        <CleanupPanel />
      </div>
    </>
  );
}
