import { describe, expect, it, vi } from 'vitest';
import { ApiError, apiPost } from './api';

describe('api errors', () => {
  it('extracts FastAPI detail instead of exposing the JSON envelope', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: 'Conflict',
      text: async () => JSON.stringify({ detail: 'rebalance analysis already running' }),
    }));

    await expect(apiPost('/api/strategy/rebalance/run')).rejects.toEqual(
      new ApiError(409, 'rebalance analysis already running'),
    );
  });
});
