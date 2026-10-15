import type { DeliveryEnvelope } from "@andarilho/scene-contract";

/** Whether the envelope's signed asset links have expired (unparseable counts as expired). */
export function envelopeExpired(
  envelope: Pick<DeliveryEnvelope, "expiresAt">,
  now = Date.now(),
): boolean {
  const expiresAt = Date.parse(envelope.expiresAt);
  return !Number.isFinite(expiresAt) || expiresAt <= now;
}

/**
 * Bounds the automatic envelope refreshes one failing viewer may ask for, so an
 * asset that keeps failing cannot refresh forever. A successful load starts the
 * count over: a later failure (another expiry, a lost graphics context) gets the
 * full budget again.
 */
export class RefreshBudget {
  private used = 0;
  private readonly limit: number;

  constructor(limit: number) {
    this.limit = limit;
  }

  /** Spends one refresh; false once the budget is used up. */
  take(): boolean {
    if (this.used >= this.limit) return false;
    this.used += 1;
    return true;
  }

  reset(): void {
    this.used = 0;
  }
}
