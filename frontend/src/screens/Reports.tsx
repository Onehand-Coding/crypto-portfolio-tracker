import { useState } from 'react';
import { Panel } from '../components/Panel';
import { Button, Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import { apiPost } from '../lib/api';
import type { GenerateExportResponse, ReportsResponse } from '../types';

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

        <Panel title={`Files (${data.files.length})`}>
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
                    <th className="text-right">Download</th>
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
                        <a href={`/api/reports/download?name=${encodeURIComponent(file.name)}`}
                           className="font-ui transition-colors"
                           style={{ color: 'var(--text-secondary)',
                                    border: '1px solid var(--border-strong)',
                                    borderRadius: 'var(--radius-control)',
                                    padding: '2px var(--space-3)', fontSize: '12px',
                                    textDecoration: 'none' }}>
                          Download
                        </a>
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
