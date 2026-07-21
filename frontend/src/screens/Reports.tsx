import { Panel } from '../components/Panel';
import { Empty, ErrorPanel, ScreenHeader } from '../components/Screen';
import { useApi } from '../lib/useApi';
import type { ReportsResponse } from '../types';

function humanSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function Reports() {
  const { data, error } = useApi<ReportsResponse>('/api/reports');

  if (error) return <ErrorPanel title="Reports" message={`Failed to load reports: ${error}`} />;
  if (!data) return <Panel title="Reports"><Empty>Loading…</Empty></Panel>;

  return (
    <>
      <ScreenHeader title="Reports & exports"
                    subtitle={`Generated files in ${data.export_dir}`} />

      <Panel title={`Files (${data.files.length})`}>
        {data.files.length === 0 ? (
          <Empty>
            No exports yet. Reports are generated from the CLI or Streamlit — this
            screen lists what exists on disk.
          </Empty>
        ) : (
          <div className="table-scroll">
            <table className="data">
              <thead>
                <tr>
                  <th className="text-left">Name</th>
                  <th className="text-right">Size</th>
                  <th className="text-left">Modified</th>
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </>
  );
}
