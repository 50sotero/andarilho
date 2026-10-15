/**
 * Spark's `SparkRenderer.dispose()` terminates its sort and LoD workers and
 * rejects the calls still running in them with "Worker terminate", which Spark
 * never catches: disposing mid-sort (a viewer rebuilt for a fresh envelope or
 * another collision source) raises an unhandled rejection. Disposal waits for
 * those calls to settle instead, up to a bound. Typed structurally so this module
 * stays free of Spark's WebGL imports.
 */

interface SparkWorkerState {
  /** Non-null while a caller holds the worker exclusively (Spark's LoD updates). */
  queue: unknown[] | null;
  /** Calls posted to the worker and not yet answered. */
  messages: Record<number, unknown>;
}

export interface SparkWorkState {
  sorting: boolean;
  sortWorker: SparkWorkerState | null;
  lodWorker: SparkWorkerState | null;
}

export const SPARK_IDLE_POLL_MS = 50;
/** A sort takes milliseconds; past this, dispose anyway rather than leak the renderer. */
export const SPARK_IDLE_MAX_WAIT_MS = 5_000;

function workerBusy(worker: SparkWorkerState | null | undefined): boolean {
  return worker != null && (worker.queue != null || Object.keys(worker.messages).length > 0);
}

/** True while a sort or LoD update is still running in one of Spark's workers. */
export function sparkBusy(spark: SparkWorkState): boolean {
  return spark.sorting || workerBusy(spark.sortWorker) || workerBusy(spark.lodWorker);
}

/** Runs `release` at once when Spark is idle, otherwise once it is (or after the bound). */
export function whenSparkIdle(spark: SparkWorkState, release: () => void): void {
  const deadline = Date.now() + SPARK_IDLE_MAX_WAIT_MS;
  const attempt = (): void => {
    if (sparkBusy(spark) && Date.now() < deadline) {
      setTimeout(attempt, SPARK_IDLE_POLL_MS);
      return;
    }
    release();
  };
  attempt();
}
