import { useState } from 'react';
import { Panel } from '../components/Panel';
import { BandMetric, KpiBand } from '../components/Band';
import { Badge, Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPost, apiPut } from '../lib/api';
import { formatPercentPlain, formatUsd } from '../lib/format';
import type {
  BackupCreateResponse, RestoreResponse, SystemHealthResponse, TargetAllocationResponse,
} from '../types';

/** Deletion outcome for a database backup. */
interface BackupDeleteResponse {
  deleted: boolean;
  name: string | null;
  error: string | null;
}

/** App + host figures. Local: the shared types file is out of scope. */
interface ResourcesResponse {
  app_version: string | null;
  python_version: string;
  cpu_percent: number | null;
  ram_percent: number | null;
  ram_used_gb: number | null;
  disk_percent: number | null;
}

interface ConnectionStatus {
  ok: boolean;
  detail: string | null;
}

interface ConnectionsResponse {
  binance: ConnectionStatus;
  coingecko: ConnectionStatus;
  btc_price_usd: number | null;
}

function formatPct1(value: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(1)}%`;
}

/** Host figures behind GET /system/resources. Nulls render as em dashes. */
function ResourcesPanel() {
  const { data } = useApi<ResourcesResponse>('/api/system/resources');
  return (
    <Panel title="Resources">
      {!data ? <Empty>Loading…</Empty> : (
        <KpiBand>
          <BandMetric label="App version" value={data.app_version || '—'} />
          <BandMetric label="Python" value={data.python_version || '—'} />
          <BandMetric label="CPU" value={formatPct1(data.cpu_percent)} />
          <BandMetric label="RAM" value={formatPct1(data.ram_percent)} />
          <BandMetric label="RAM used"
                      value={data.ram_used_gb === null ? '—' : `${data.ram_used_gb.toFixed(1)} GB`} />
          <BandMetric label="Disk" value={formatPct1(data.disk_percent)} />
        </KpiBand>
      )}
    </Panel>
  );
}

/** Live Binance + CoinGecko probe. POST: it touches the network. */
function ConnectionsPanel() {
  const [result, setResult] = useState<ConnectionsResponse | null>(null);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function run() {
    setTesting(true);
    setMessage(null);
    try {
      setResult(await apiPost<ConnectionsResponse>('/api/system/connections'));
    } catch (e) {
      setMessage(`Connection test failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTesting(false);
    }
  }

  return (
    <Panel title="Connections">
      <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)',
                                                             marginBottom: 'var(--space-3)' }}>
        <span className="font-ui" style={{ color: message ? 'var(--negative)' : 'var(--text-secondary)',
                                           fontSize: '13px' }}>
          {message ?? 'Probe Binance and CoinGecko over the network.'}
        </span>
        <Button variant="secondary" onClick={run} disabled={testing}>
          {testing ? 'Testing…' : 'Run connection test'}
        </Button>
      </div>
      {result && (
        <>
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-3)' }}>
            <Badge text={result.binance.ok ? 'BINANCE OK' : 'BINANCE FAILED'}
                   tone={result.binance.ok ? 'positive' : 'negative'} />
            <Badge text={result.coingecko.ok ? 'COINGECKO OK' : 'COINGECKO FAILED'}
                   tone={result.coingecko.ok ? 'positive' : 'negative'} />
            <span className="font-mono" style={{ fontSize: '13px' }}>
              BTC {formatUsd(result.btc_price_usd)}
            </span>
          </div>
          {[result.binance.detail, result.coingecko.detail].some(Boolean) && (
            <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                            marginBottom: 0 }}>
              {[result.binance.detail, result.coingecko.detail].filter(Boolean).join(' · ')}
            </p>
          )}
        </>
      )}
    </Panel>
  );
}

interface DraftRow { symbol: string; pct: string }

/** Editable target allocation. Reads fractions, edits percentages, writes back. */
function TargetAllocationPanel({
  allocation, onSaved,
}: { allocation: Record<string, number>; onSaved: () => void }) {
  const entries = Object.entries(allocation).sort((a, b) => b[1] - a[1]);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<DraftRow[]>([]);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  function begin() {
    setDraft(entries.map(([symbol, weight]) => ({
      symbol, pct: String(+(weight * 100).toFixed(4)),
    })));
    setMessage(null);
    setEditing(true);
  }

  const draftSum = draft.reduce((sum, r) => sum + (Number(r.pct) || 0), 0);
  const valid = draft.every((r) => r.symbol.trim() && Number.isFinite(Number(r.pct))
    && Number(r.pct) >= 0);

  async function save() {
    setSaving(true);
    setMessage(null);
    try {
      const payload: Record<string, number> = {};
      for (const row of draft) payload[row.symbol.trim().toUpperCase()] = Number(row.pct) / 100;
      const result = await apiPut<TargetAllocationResponse>(
        '/api/system/target-allocation', { allocation: payload },
      );
      setEditing(false);
      setMessage(result.sums_to_one
        ? 'Target allocation saved.'
        : `Saved, but weights sum to ${formatPercentPlain(result.sum * 100)}, not 100%.`);
      onSaved();
    } catch (e) {
      setMessage(`Save failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Panel title="Target allocation">
      <div className="flex flex-wrap items-center justify-between"
           style={{ gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
        <span className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
          {message ?? 'Drives rebalancing, DCA and every drift figure. Weights are percentages.'}
        </span>
        {!editing && <Button variant="secondary" onClick={begin}>Edit allocation</Button>}
      </div>

      {!editing ? (
        entries.length === 0 ? (
          <Empty>No target allocation configured. Use “Edit allocation” to set one.</Empty>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th className="text-left">Asset</th>
                  <th className="text-right">Target</th>
                  <th className="text-left">Weight</th>
                </tr>
              </thead>
              <tbody>
                {entries.map(([symbol, weight]) => (
                  <tr key={symbol}>
                    <td className="text-left" style={{ fontWeight: 500 }}>{symbol}</td>
                    <td className="text-right">{formatPercentPlain(weight * 100)}</td>
                    <td className="text-left">
                      <div style={{ height: '8px', width: '200px', background: 'var(--surface-2)',
                                    borderRadius: '2px' }}>
                        <div style={{ height: '100%', width: `${weight * 100 * 2}%`,
                                      maxWidth: '100%', background: 'var(--action)',
                                      borderRadius: '2px' }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : (
        <div className="flex flex-col" style={{ gap: 'var(--space-2)' }}>
          {draft.map((row, i) => (
            <div key={i} className="flex items-center" style={{ gap: 'var(--space-3)' }}>
              <input
                value={row.symbol}
                onChange={(e) => setDraft((d) => d.map((r, j) =>
                  j === i ? { ...r, symbol: e.target.value } : r))}
                placeholder="SYMBOL"
                className="font-mono"
                style={{ background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                         borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                         padding: 'var(--space-1) var(--space-3)', width: '120px', fontSize: '13px',
                         textTransform: 'uppercase' }}
              />
              <input
                value={row.pct}
                onChange={(e) => setDraft((d) => d.map((r, j) =>
                  j === i ? { ...r, pct: e.target.value } : r))}
                inputMode="decimal"
                className="font-mono"
                style={{ background: 'var(--surface-0)', border: '1px solid var(--border-strong)',
                         borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                         padding: 'var(--space-1) var(--space-3)', width: '90px', fontSize: '13px',
                         textAlign: 'right' }}
              />
              <span className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>%</span>
              <button
                onClick={() => setDraft((d) => d.filter((_, j) => j !== i))}
                className="font-ui transition-colors"
                style={{ background: 'transparent', color: 'var(--text-tertiary)',
                         border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                         padding: 'var(--space-1) var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
              >
                Remove
              </button>
            </div>
          ))}

          <div className="flex items-center justify-between"
               style={{ gap: 'var(--space-3)', marginTop: 'var(--space-2)' }}>
            <button
              onClick={() => setDraft((d) => [...d, { symbol: '', pct: '0' }])}
              className="font-ui transition-colors"
              style={{ background: 'transparent', color: 'var(--text-secondary)',
                       border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-control)',
                       padding: 'var(--space-1) var(--space-3)', fontSize: '13px', cursor: 'pointer' }}
            >
              + Add asset
            </button>
            <span className="font-mono" style={{ fontSize: '13px',
                     color: Math.abs(draftSum - 100) < 0.01 ? 'var(--positive)' : 'var(--warning)' }}>
              Sum: {draftSum.toFixed(2)}%
            </span>
          </div>

          <div className="flex items-center" style={{ gap: 'var(--space-3)',
                                                      marginTop: 'var(--space-3)' }}>
            <Button onClick={save} disabled={saving || !valid}>
              {saving ? 'Saving…' : 'Save allocation'}
            </Button>
            <Button variant="secondary" onClick={() => setEditing(false)} disabled={saving}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </Panel>
  );
}

function humanSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function humanAge(seconds: number | null): string {
  if (seconds === null) return 'never';
  const minutes = Math.round(seconds / 60);
  if (minutes < 1) return 'just now';
  if (minutes < 60) return `${minutes}m ago`;
  return `${Math.round(minutes / 60)}h ago`;
}

export function SystemHealth() {
  const { data, error, reload } = useApi<SystemHealthResponse>('/api/system/health');
  const [backingUp, setBackingUp] = useState(false);
  const [backupMsg, setBackupMsg] = useState<string | null>(null);
  const [confirmRestore, setConfirmRestore] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);
  const [restoreMsg, setRestoreMsg] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  async function restore(name: string) {
    setRestoring(true);
    setRestoreMsg(null);
    try {
      const result = await apiPost<RestoreResponse>('/api/system/restore', { name });
      setRestoreMsg(result.restored
        ? `Restored from ${name}. The prior database was saved as ${result.safety_backup}.`
        : `Restore failed: ${result.error ?? 'unknown error'}`);
      setConfirmRestore(null);
      if (result.restored) reload();
    } catch (e) {
      setRestoreMsg(`Restore failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRestoring(false);
    }
  }

  async function deleteBackup(name: string) {
    setDeleting(true);
    try {
      const result = await apiPost<BackupDeleteResponse>('/api/system/backup/delete', {
        name, confirm: true,
      });
      setBackupMsg(result.deleted ? `Backup deleted: ${name}.`
                                  : `Nothing deleted${result.error ? `: ${result.error}` : '.'}`);
      setConfirmDelete(null);
      if (result.deleted) reload();
    } catch (e) {
      setBackupMsg(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDeleting(false);
    }
  }

  async function createBackup() {
    setBackingUp(true);
    setBackupMsg(null);
    try {
      const result = await apiPost<BackupCreateResponse>('/api/system/backup');
      setBackupMsg(result.created ? `Backup created: ${result.name}`
                                  : `Backup failed: ${result.error ?? 'unknown error'}`);
      if (result.created) reload();
    } catch (e) {
      setBackupMsg(`Backup failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBackingUp(false);
    }
  }

  if (error) return <ErrorPanel title="System" message={`Failed to load system health: ${error}`} />;
  if (!data) return <Panel title="System"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader title="System & settings"
                    subtitle="Environment, database and configuration" />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Environment">
          <div className="flex items-center" style={{ gap: 'var(--space-3)',
                                                      marginBottom: 'var(--space-4)' }}>
            <Badge text={data.environment_label}
                   tone={data.is_testnet ? 'warning' : 'action'} />
            <Badge text={data.binance_configured ? 'API KEYS PRESENT' : 'API KEYS MISSING'}
                   tone={data.binance_configured ? 'positive' : 'negative'} />
            <Badge text={data.live_trading_enabled ? 'LIVE TRADING ENABLED' : 'LIVE TRADING OFF'}
                   tone={data.live_trading_enabled ? 'negative' : 'neutral'} />
          </div>
          <p className="font-mono" style={{ color: 'var(--text-secondary)', fontSize: '12px',
                                            margin: 0, wordBreak: 'break-all' }}>
            {data.database_path}
          </p>
          {!data.database_exists && (
            <p className="font-ui text-sm" style={{ color: 'var(--negative)',
                                                    marginTop: 'var(--space-3)', marginBottom: 0 }}>
              Database file does not exist at this path.
            </p>
          )}
        </Panel>

        <Panel title="Database">
          <KpiBand>
            <BandMetric label="Transactions" value={data.transaction_count.toLocaleString()} />
            <BandMetric label="Assets" value={String(data.asset_count)} />
            <BandMetric label="Snapshots" value={String(data.snapshot_count)} />
            <BandMetric label="Size" value={humanSize(data.database_size_bytes)} />
            <BandMetric label="Metrics cache"
                        value={humanAge(data.metrics_cache_age_seconds)} />
          </KpiBand>
        </Panel>

        <TargetAllocationPanel allocation={data.target_allocation} onSaved={reload} />

        <ResourcesPanel />

        <ConnectionsPanel />

        <Panel title="Trading limits">
          <KpiBand>
            <BandMetric label="Minimum trade" value={formatUsd(data.minimum_trade_usd)} />
          </KpiBand>
        </Panel>

        <Panel title={`Backups (${data.backups.length})`}>
          <div className="flex flex-wrap items-center justify-between"
               style={{ gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
            <span className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
              {backupMsg ?? 'Create an on-demand copy of the database. This only reads the '
                + 'live file — it never modifies it.'}
            </span>
            <Button variant="secondary" onClick={createBackup} disabled={backingUp}>
              {backingUp ? 'Creating…' : 'Create backup'}
            </Button>
          </div>
          {restoreMsg && (
            <p className="font-ui" style={{ fontSize: '13px', marginTop: 0,
                     marginBottom: 'var(--space-3)',
                     color: restoreMsg.startsWith('Restore failed')
                       ? 'var(--negative)' : 'var(--text-secondary)' }}>
              {restoreMsg}
            </p>
          )}
          {data.backups.length === 0 ? (
            <Empty>No database backups found.</Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Name</th>
                    <th className="text-right">Size</th>
                    <th className="text-left">Created</th>
                    <th className="text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {data.backups.map((backup) => (
                    <tr key={backup.name}>
                      <td className="text-left">{backup.name}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {humanSize(backup.size_bytes)}
                      </td>
                      <td className="text-left" style={{ color: 'var(--text-tertiary)' }}>
                        {backup.modified.slice(0, 16).replace('T', ' ')}
                      </td>
                      <td className="text-right">
                        <span className="flex items-center justify-end" style={{ gap: 'var(--space-2)' }}>
                          <a href={`/api/system/backup/download?name=${encodeURIComponent(backup.name)}`}
                             className="font-ui transition-colors"
                             style={{ color: 'var(--text-secondary)',
                                      border: '1px solid var(--border-strong)',
                                      borderRadius: 'var(--radius-control)',
                                      padding: '2px var(--space-3)', fontSize: '12px',
                                      textDecoration: 'none' }}>
                            Download
                          </a>
                          {confirmDelete === backup.name ? (
                            <>
                              <button
                                onClick={() => deleteBackup(backup.name)}
                                disabled={deleting}
                                className="font-ui transition-colors"
                                style={{ background: 'color-mix(in srgb, var(--negative) 18%, transparent)',
                                         color: 'var(--negative)',
                                         border: '1px solid color-mix(in srgb, var(--negative) 35%, transparent)',
                                         borderRadius: 'var(--radius-control)',
                                         padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                              >
                                {deleting ? '…' : 'Confirm'}
                              </button>
                              <button
                                onClick={() => setConfirmDelete(null)}
                                className="font-ui transition-colors"
                                style={{ background: 'transparent', color: 'var(--text-tertiary)',
                                         border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                                         padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                              >
                                Cancel
                              </button>
                            </>
                          ) : (
                            <button
                              onClick={() => { setConfirmDelete(backup.name); setBackupMsg(null); }}
                              className="font-ui transition-colors"
                              style={{ background: 'transparent', color: 'var(--text-secondary)',
                                       border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-control)',
                                       padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                            >
                              Delete
                            </button>
                          )}
                          {confirmRestore === backup.name ? (
                          // Restore overwrites the live database, so it is a
                          // deliberate two-step: the current DB is snapshotted
                          // first, but the confirm is still explicit.
                          <span className="flex items-center justify-end" style={{ gap: 'var(--space-2)' }}>
                            <span className="font-ui" style={{ color: 'var(--warning)', fontSize: '11px' }}>
                              Overwrite current DB?
                            </span>
                            <button
                              onClick={() => restore(backup.name)}
                              disabled={restoring}
                              className="font-ui transition-colors"
                              style={{ background: 'color-mix(in srgb, var(--negative) 18%, transparent)',
                                       color: 'var(--negative)',
                                       border: '1px solid color-mix(in srgb, var(--negative) 35%, transparent)',
                                       borderRadius: 'var(--radius-control)',
                                       padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                            >
                              {restoring ? 'Restoring…' : 'Confirm'}
                            </button>
                            <button
                              onClick={() => setConfirmRestore(null)}
                              className="font-ui transition-colors"
                              style={{ background: 'transparent', color: 'var(--text-tertiary)',
                                       border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                                       padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                            >
                              Cancel
                            </button>
                          </span>
                        ) : (
                          <button
                            onClick={() => { setConfirmRestore(backup.name); setRestoreMsg(null); }}
                            className="font-ui transition-colors"
                            style={{ background: 'transparent', color: 'var(--text-secondary)',
                                     border: '1px solid var(--border-strong)', borderRadius: 'var(--radius-control)',
                                     padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                          >
                            Restore
                          </button>
                        )}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>
    </>
  );
}
