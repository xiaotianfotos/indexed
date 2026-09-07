/** A cancelled transport may still resolve; only the newest request may render. */
export class LatestRequest {
  #controller: AbortController | null = null;
  #generation = 0;

  begin(): { signal: AbortSignal; current: () => boolean } {
    this.cancel();
    const generation = this.#generation;
    const controller = new AbortController();
    this.#controller = controller;
    return { signal: controller.signal, current: () => generation === this.#generation && !controller.signal.aborted };
  }

  cancel(): void {
    this.#generation += 1;
    this.#controller?.abort();
    this.#controller = null;
  }
}
