const previous = new WeakMap<HTMLElement, HTMLElement | null>();
const wired = new WeakSet<HTMLElement>();
const focusable = 'button, input, textarea, select, a[href], [tabindex]:not([tabindex="-1"])';

export function openDialog(dialog: HTMLElement, initial: HTMLElement, fallback?: HTMLElement): void {
  const active = document.activeElement instanceof HTMLElement ? document.activeElement : null;
  previous.set(dialog, active?.getClientRects().length ? active : fallback || null);
  dialog.hidden = false;
  const shell = document.querySelector<HTMLElement>(".app-shell");
  if (shell) shell.inert = true;
  if (!wired.has(dialog)) {
    wired.add(dialog);
    dialog.addEventListener("keydown", event => {
      if (event.key !== "Tab") return;
      const elements = [...dialog.querySelectorAll<HTMLElement>(focusable)]
        .filter(element => element.getClientRects().length > 0 && !element.matches(":disabled, [inert]"));
      const first = elements[0];
      const last = elements.at(-1);
      if (!first || !last) { event.preventDefault(); dialog.focus(); return; }
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialog)) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
    });
  }
  initial.focus();
}

export function closeDialog(dialog: HTMLElement): void {
  if (dialog.hidden) return;
  dialog.hidden = true;
  const shell = document.querySelector<HTMLElement>(".app-shell");
  if (shell) shell.inert = Boolean(document.querySelector('[role="dialog"]:not([hidden])'));
  const target = previous.get(dialog);
  if (target?.isConnected && target.getClientRects().length) target.focus();
  else document.querySelector<HTMLElement>("#view-toggle")?.focus();
  previous.delete(dialog);
}
