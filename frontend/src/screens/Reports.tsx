import { useEffect, useState } from 'react';
import { Panel } from '../components/Panel';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiGet, apiPost } from '../lib/api';
import { NULL_GLYPH } from '../lib/format';
import type { GenerateExportResponse, ReportsResponse } from '../types';

/** One generated export, for on-screen preview. Tabular files arrive as
 *  columns plus rows; HTML/JSON/images are flagged for their viewer. */
interface ReportPreview {
  name: string;
  lines: string[];
  truncated: boolean;
  total_lines: number;
  kind: 'text' | 'table' | 'html' | 'json' | 'image';
  columns: string[];
  rows: (string | number | boolean | null)[][];
  image_url: string | null;
}

function downloadUrl(name: string): string {
  return `/api/reports/download?name=${encodeURIComponent(name)}`;
}

/** File preview in a modal dialog -- the Drive/Dropbox/GitHub pattern. The
 *  table below the fold made the old inline panel look dead on click. */
function PreviewModal({ preview, onClose }: {
  preview: ReportPreview; onClose: () => void;
}) {
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  let body: React.ReactNode;
  if (preview.kind === 'table') {
    body = (
      <>
        <div className="table-scroll">
          <table className="data">
            <thead>
              <tr>
                {preview.columns.map((col) => (
                  <th key={col} className="text-left">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j} className="text-left"
                        style={cell === null ? { color: 'var(--text-tertiary)' } : undefined}>
                      {cell === null ? NULL_GLYPH : String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {preview.truncated && (
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginBottom: 0, marginTop: 'var(--space-2)' }}>
            …{preview.total_lines - preview.rows.length} more rows
          </p>
        )}
      </>
    );
  } else if (preview.kind === 'html') {
    // Sandboxed and origin-less, so embedded scripts cannot run -- the
    // exports are style-only templates, which render faithfully anyway.
    body = (
      <iframe sandbox="" title={`Preview of ${preview.name}`}
              src={downloadUrl(preview.name)}
              style={{ width: '100%', height: '60vh', background: '#fff',
                       border: '1px solid var(--border)',
                       borderRadius: 'var(--radius-control)' }} />
    );
  } else if (preview.kind === 'json') {
    let pretty = preview.lines.join('\n');
    try {
      pretty = JSON.stringify(JSON.parse(pretty), null, 2);
    } catch {
      // Not parseable JSON: show the raw lines rather than failing.
    }
    body = (
      <>
        <pre className="font-mono"
             style={{ background: 'var(--surface-0)', border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                      padding: 'var(--space-3)', fontSize: '12px', overflowX: 'auto',
                      whiteSpace: 'pre-wrap', margin: 0 }}>
          {pretty}
        </pre>
        {preview.truncated && (
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginBottom: 0, marginTop: 'var(--space-2)' }}>
            …{preview.total_lines - preview.lines.length} more lines
          </p>
        )}
      </>
    );
  } else if (preview.kind === 'image' && preview.image_url) {
    body = (
      <img src={preview.image_url} alt={`Preview of ${preview.name}`}
           style={{ maxWidth: '100%', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-control)' }} />
    );
  } else {
    body = (
      <>
        <pre className="font-mono"
             style={{ background: 'var(--surface-0)', border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-control)', color: 'var(--text-primary)',
                      padding: 'var(--space-3)', fontSize: '12px', overflowX: 'auto',
                      whiteSpace: 'pre-wrap', margin: 0 }}>
          {preview.lines.join('\n')}
        </pre>
        {preview.truncated && (
          <p className="font-ui" style={{ color: 'var(--text-tertiary)', fontSize: '12px',
                                          marginBottom: 0, marginTop: 'var(--space-2)' }}>
            …{preview.total_lines - preview.lines.length} more lines
          </p>
        )}
      </>
    );
  }

  return (
    <div role="dialog" aria-modal="true" aria-label={`Preview of ${preview.name}`}
         onClick={onClose}
         style={{ position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.65)',
                  zIndex: 50, display: 'flex', alignItems: 'center',
                  justifyContent: 'center', padding: 'var(--space-5)' }}>
      <div onClick={(e) => e.stopPropagation()}
           style={{ background: 'var(--surface-1)', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-panel)', padding: 'var(--space-4)',
                    maxWidth: '960px', width: '100%', maxHeight: '85vh',
                    display: 'flex', flexDirection: 'column' }}>
        <div className="flex items-center justify-between"
             style={{ gap: 'var(--space-3)', marginBottom: 'var(--space-3)' }}>
          <span className="font-ui" style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
            Preview: {preview.name}
          </span>
          <button
            onClick={onClose}
            className="font-ui transition-colors"
            style={{ background: 'transparent', color: 'var(--text-tertiary)',
                     border: '1px solid var(--border)', borderRadius: 'var(--radius-control)',
                     padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
          >
            Close
          </button>
        </div>
        <div style={{ overflow: 'auto' }}>
          {body}
        </div>
      </div>
    </div>
  );
}

/** Deletion outcome for a generated export. */
interface DeleteResult {
  deleted: boolean;
  name: string | null;
  error: string | null;
}

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

const DATA_TYPES = [
  { id: 'transactions', label: 'Transactions' },
  { id: 'holdings', label: 'Holdings' },
];
const FORMATS = [
  { id: 'csv', label: 'CSV' },
  { id: 'excel', label: 'Excel' },
];
const SUMMARY_FORMATS = [
  { id: 'csv', label: 'CSV' },
  { id: 'excel', label: 'Excel' },
  { id: 'html', label: 'HTML' },
];
const TREND_TIMEFRAMES = [
  { id: 'long_term', label: 'Long term' },
  { id: 'swing', label: 'Swing' },
  { id: 'day', label: 'Day' },
];
const TREND_FORMATS = [
  { id: 'csv', label: 'CSV' },
  { id: 'json', label: 'JSON' },
  { id: 'html', label: 'HTML' },
];

function Choice({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button onClick={onClick} className="font-ui transition-colors"
      style={{ background: active ? 'var(--surface-2)' : 'transparent',
               color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
               border: `1px solid ${active ? 'var(--border-strong)' : 'var(--border)'}`,
               borderRadius: 'var(--radius-control)',
               padding: 'var(--space-2) var(--space-3)', fontSize: '13px', cursor: 'pointer' }}>
      {children}
    </button>
  );
}

export function Reports() {
  const { data, error, reload } = useApi<ReportsResponse>('/api/reports');
  const [dataType, setDataType] = useState('transactions');
  const [format, setFormat] = useState('csv');
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [summaryFormat, setSummaryFormat] = useState('csv');
  const [summaryBusy, setSummaryBusy] = useState(false);
  const [summaryMsg, setSummaryMsg] = useState<string | null>(null);
  const [trendTimeframe, setTrendTimeframe] = useState('long_term');
  const [trendFormat, setTrendFormat] = useState('csv');
  const [trendBusy, setTrendBusy] = useState(false);
  const [trendMsg, setTrendMsg] = useState<string | null>(null);
  const [chartsBusy, setChartsBusy] = useState(false);
  const [chartsMsg, setChartsMsg] = useState<string | null>(null);
  const [preview, setPreview] = useState<ReportPreview | null>(null);
  const [previewBusy, setPreviewBusy] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [fileMsg, setFileMsg] = useState<string | null>(null);

  async function generate() {
    setGenerating(true);
    setMessage(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/generate', {
        data_type: dataType, format,
      });
      setMessage(`Generated ${res.name}.`);
      reload();
    } catch (e) {
      setMessage(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setGenerating(false);
    }
  }

  async function generateSummary() {
    setSummaryBusy(true);
    setSummaryMsg(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/summary', {
        format: summaryFormat,
      });
      setSummaryMsg(`Generated ${res.name}.`);
      reload();
    } catch (e) {
      setSummaryMsg(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setSummaryBusy(false);
    }
  }

  async function generateTrend() {
    setTrendBusy(true);
    setTrendMsg(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/trend', {
        timeframe: trendTimeframe, format: trendFormat,
      });
      setTrendMsg(`Generated ${res.name}.`);
      reload();
    } catch (e) {
      setTrendMsg(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTrendBusy(false);
    }
  }

  async function generateCharts() {
    setChartsBusy(true);
    setChartsMsg(null);
    try {
      const res = await apiPost<GenerateExportResponse>('/api/reports/charts');
      setChartsMsg(`Generated ${res.name}.`);
      reload();
    } catch (e) {
      setChartsMsg(`Export failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setChartsBusy(false);
    }
  }

  async function showPreview(name: string) {
    setPreviewBusy(name);
    setPreviewError(null);
    try {
      const res = await apiGet<ReportPreview>(
        `/api/reports/preview?name=${encodeURIComponent(name)}`);
      setPreview(res);
    } catch (e) {
      setPreview(null);
      setPreviewError(`Preview failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPreviewBusy(null);
    }
  }

  async function del(name: string) {
    setDeleteBusy(true);
    setFileMsg(null);
    try {
      const res = await apiPost<DeleteResult>('/api/reports/delete', {
        name, confirm: true,
      });
      setFileMsg(res.deleted ? `Deleted ${name}.`
                             : `Nothing deleted${res.error ? `: ${res.error}` : '.'}`);
      setDeleteConfirm(null);
      if (preview?.name === name) setPreview(null);
      if (res.deleted) reload();
    } catch (e) {
      setFileMsg(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDeleteBusy(false);
    }
  }

  if (error) return <ErrorPanel title="Reports" message={`Failed to load reports: ${error}`} />;
  if (!data) return <Panel title="Reports"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader title="Reports & exports" subtitle={`Generated files in ${data.export_dir}`} />

      <div className="flex flex-col" style={{ gap: 'var(--space-3)' }}>
        <Panel title="Generate export">
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-4)' }}>
            <div className="flex" style={{ gap: 'var(--space-2)' }}>
              {DATA_TYPES.map((d) => (
                <Choice key={d.id} active={dataType === d.id} onClick={() => setDataType(d.id)}>
                  {d.label}
                </Choice>
              ))}
            </div>
            <div className="flex" style={{ gap: 'var(--space-2)' }}>
              {FORMATS.map((f) => (
                <Choice key={f.id} active={format === f.id} onClick={() => setFormat(f.id)}>
                  {f.label}
                </Choice>
              ))}
            </div>
            <Button onClick={generate} disabled={generating}>
              {generating ? 'Generating…' : 'Generate'}
            </Button>
            {message && (
              <span className="font-ui" style={{ fontSize: '13px',
                       color: message.startsWith('Export failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
                {message}
              </span>
            )}
          </div>
        </Panel>

        <Panel title="Portfolio summary">
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-4)' }}>
            <div className="flex" style={{ gap: 'var(--space-2)' }}>
              {SUMMARY_FORMATS.map((f) => (
                <Choice key={f.id} active={summaryFormat === f.id}
                        onClick={() => setSummaryFormat(f.id)}>
                  {f.label}
                </Choice>
              ))}
            </div>
            <Button onClick={generateSummary} disabled={summaryBusy}>
              {summaryBusy ? 'Generating…' : 'Generate summary'}
            </Button>
            {summaryMsg && (
              <span className="font-ui" style={{ fontSize: '13px',
                       color: summaryMsg.startsWith('Export failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
                {summaryMsg}
              </span>
            )}
          </div>
        </Panel>

        <Panel title="Trend report">
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-4)' }}>
            <div className="flex" style={{ gap: 'var(--space-2)' }}>
              {TREND_TIMEFRAMES.map((t) => (
                <Choice key={t.id} active={trendTimeframe === t.id}
                        onClick={() => setTrendTimeframe(t.id)}>
                  {t.label}
                </Choice>
              ))}
            </div>
            <div className="flex" style={{ gap: 'var(--space-2)' }}>
              {TREND_FORMATS.map((f) => (
                <Choice key={f.id} active={trendFormat === f.id}
                        onClick={() => setTrendFormat(f.id)}>
                  {f.label}
                </Choice>
              ))}
            </div>
            <Button onClick={generateTrend} disabled={trendBusy}>
              {trendBusy ? 'Generating…' : 'Generate trend'}
            </Button>
            {trendMsg && (
              <span className="font-ui" style={{ fontSize: '13px',
                       color: trendMsg.startsWith('Export failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
                {trendMsg}
              </span>
            )}
          </div>
        </Panel>

        <Panel title="Charts">
          <div className="flex flex-wrap items-center" style={{ gap: 'var(--space-4)' }}>
            <span className="font-ui" style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
              Allocation, P/L and value history as PNG files.
            </span>
            <Button onClick={generateCharts} disabled={chartsBusy}>
              {chartsBusy ? 'Generating…' : 'Generate charts'}
            </Button>
            {chartsMsg && (
              <span className="font-ui" style={{ fontSize: '13px',
                       color: chartsMsg.startsWith('Export failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
                {chartsMsg}
              </span>
            )}
          </div>
        </Panel>

        <Panel title={`Files (${data.files.length})`}>
          {fileMsg && (
            <p className="font-ui" style={{ fontSize: '13px', margin: '0 0 var(--space-3) 0',
                     color: fileMsg.startsWith('Delete failed') ? 'var(--negative)' : 'var(--text-secondary)' }}>
              {fileMsg}
            </p>
          )}
          {data.files.length === 0 ? (
            <Empty>No exports yet. Generate one above.</Empty>
          ) : (
            <div className="table-scroll">
              <table className="data">
                <thead>
                  <tr>
                    <th className="text-left">Name</th>
                    <th className="text-right">Size</th>
                    <th className="text-left">Modified</th>
                    <th className="text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {data.files.map((file) => (
                    <tr key={file.path}>
                      <td className="text-left">{file.name}</td>
                      <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
                        {humanSize(file.size_bytes)}
                      </td>
                      <td className="text-left" style={{ color: 'var(--text-tertiary)' }}>
                        {file.modified.slice(0, 16).replace('T', ' ')}
                      </td>
                      <td className="text-right">
                        <span className="flex items-center justify-end"
                              style={{ gap: 'var(--space-2)' }}>
                          <a href={`/api/reports/download?name=${encodeURIComponent(file.name)}`}
                             className="font-ui transition-colors"
                             style={{ color: 'var(--text-secondary)',
                                      border: '1px solid var(--border-strong)',
                                      borderRadius: 'var(--radius-control)',
                                      padding: '2px var(--space-3)', fontSize: '12px',
                                      textDecoration: 'none' }}>
                            Download
                          </a>
                          <button
                            onClick={() => showPreview(file.name)}
                            disabled={previewBusy === file.name}
                            className="font-ui transition-colors"
                            style={{ background: 'transparent', color: 'var(--text-secondary)',
                                     border: '1px solid var(--border-strong)',
                                     borderRadius: 'var(--radius-control)',
                                     padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                          >
                            {previewBusy === file.name ? '…' : 'Preview'}
                          </button>
                          {deleteConfirm === file.name ? (
                            <>
                              <button
                                onClick={() => del(file.name)}
                                disabled={deleteBusy}
                                className="font-ui transition-colors"
                                style={{ background: 'color-mix(in srgb, var(--negative) 18%, transparent)',
                                         color: 'var(--negative)',
                                         border: '1px solid color-mix(in srgb, var(--negative) 35%, transparent)',
                                         borderRadius: 'var(--radius-control)',
                                         padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                              >
                                {deleteBusy ? '…' : 'Confirm'}
                              </button>
                              <button
                                onClick={() => setDeleteConfirm(null)}
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
                              onClick={() => { setDeleteConfirm(file.name); setFileMsg(null); }}
                              className="font-ui transition-colors"
                              style={{ background: 'transparent', color: 'var(--text-secondary)',
                                       border: '1px solid var(--border-strong)',
                                       borderRadius: 'var(--radius-control)',
                                       padding: '2px var(--space-3)', fontSize: '12px', cursor: 'pointer' }}
                            >
                              Delete
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
          {previewError && (
            <p className="font-ui" style={{ fontSize: '13px', marginBottom: 0,
                     marginTop: 'var(--space-3)', color: 'var(--negative)' }}>
              {previewError}
            </p>
          )}
        </Panel>
        {preview && <PreviewModal preview={preview} onClose={() => setPreview(null)} />}
      </div>
    </>
  );
}
