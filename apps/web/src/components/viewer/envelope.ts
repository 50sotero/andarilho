import type { DeliveryEnvelope } from "@andarilho/scene-contract";

/** Whether the envelope's signed asset links have expired (unparseable counts as expired). */
export function envelopeExpired(
  envelope: Pick<DeliveryEnvelope, "expiresAt">,
  now = Date.now(),
): boolean {
  const expiresAt = Date.parse(envelope.expiresAt);
  return !Number.isFinite(expiresAt) || expiresAt <= now;
}
