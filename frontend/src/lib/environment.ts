/**
 * Refetch signal for the app shell's environment badges.
 *
 * App fetches /api/portfolio/cockpit once on mount, so after Settings
 * persists a testnet_mode flip the TESTNET/LIVE badges (and the database
 * path) would otherwise stay frozen until a full page reload -- while the
 * backend, the caches, and every freshly mounted screen are already on the
 * other environment. Settings.save() dispatches this after a changed save;
 * App listens and refetches. A plain DOM event keeps the checklist of
 * screens decoupled from the shell: no context, no store.
 */
export const ENVIRONMENT_CHANGED_EVENT = 'environment-changed';

export function notifyEnvironmentChanged(): void {
  window.dispatchEvent(new Event(ENVIRONMENT_CHANGED_EVENT));
}
