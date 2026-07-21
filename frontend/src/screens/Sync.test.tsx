import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Sync } from './Sync';

class FakeEventSource {
  static instances: FakeEventSource[] = [];
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  url: string;

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  emit(payload: object) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  emitRaw(data: string) {
    this.onmessage?.({ data });
  }

  triggerError() {
    this.onerror?.();
  }
}

beforeEach(() => {
  vi.unstubAllGlobals();
  FakeEventSource.instances = [];
  vi.stubGlobal('EventSource', FakeEventSource);
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true, json: async () => ({ status: 'started' }),
  }));
});

describe('Sync', () => {
  it('streams progress lines from the core rather than a blank spinner', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'progress', message: 'Fetching chunk 1 of 3' });

    await waitFor(() => expect(screen.getByText('Fetching chunk 1 of 3')).toBeDefined());
  });

  it('surfaces errors instead of appearing to succeed', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'error', message: 'binance unreachable' });

    await waitFor(() => expect(screen.getByText(/binance unreachable/)).toBeDefined());
  });

  it('closes the stream once complete', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'complete', message: 'Sync complete' });

    await waitFor(() => expect(FakeEventSource.instances[0].closed).toBe(true));
  });
});

describe('Sync connection loss', () => {
  it('surfaces a lost-connection message and re-enables the button on a dropped stream', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'progress', message: 'Fetching chunk 1 of 3' });
    await waitFor(() => expect(screen.getByText('Fetching chunk 1 of 3')).toBeDefined());

    FakeEventSource.instances[0].triggerError();

    await waitFor(() => expect(screen.getByText(/Lost connection/i)).toBeDefined());
    expect(screen.getByRole('button', { name: /start sync/i })).toBeDefined();
  });

  it('does not report a false error when onerror fires after a successful complete', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emit({ event: 'complete', message: 'Sync complete' });
    await waitFor(() => expect(screen.getByText('Sync complete')).toBeDefined());

    // This mirrors what a real EventSource does: the server closes the
    // stream after a terminal event, which the browser also reports via
    // onerror.
    FakeEventSource.instances[0].triggerError();

    expect(screen.queryByText(/Lost connection/i)).toBeNull();
  });

  it('surfaces an error and does not crash on a malformed event payload', async () => {
    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    FakeEventSource.instances[0].emitRaw('not valid json{{{');

    await waitFor(() => expect(screen.getByText(/unreadable event/i)).toBeDefined());
    // The screen must still be alive -- e.g. the Sync panel heading.
    expect(screen.getByText(/Sync is the only action/i)).toBeDefined();
  });

  it('closes the EventSource when the component unmounts mid-sync', async () => {
    const { unmount } = render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => expect(FakeEventSource.instances.length).toBe(1));
    unmount();

    expect(FakeEventSource.instances[0].closed).toBe(true);
  });
});

describe('Sync start failure', () => {
  it('surfaces a visible error when POST /api/sync itself fails, not a stuck "Syncing…" state', async () => {
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false, status: 409, text: async () => 'A sync is already running',
    }));

    render(<Sync />);
    fireEvent.click(screen.getByRole('button', { name: /start sync/i }));

    await waitFor(() => {
      expect(screen.getByText(/already running/i)).toBeDefined();
    });

    // Must not be left stuck on the disabled "Syncing…" label, and must not
    // have opened a stream since the POST never succeeded.
    expect(screen.getByRole('button', { name: /start sync/i })).toBeDefined();
    expect(FakeEventSource.instances.length).toBe(0);
  });
});
