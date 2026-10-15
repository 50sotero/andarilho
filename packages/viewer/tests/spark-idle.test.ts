import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SPARK_IDLE_MAX_WAIT_MS,
  SPARK_IDLE_POLL_MS,
  sparkBusy,
  whenSparkIdle,
  type SparkWorkState,
} from "../src/render/spark-idle";

const idleWorker = () => ({ queue: null, messages: {} });

function sparkState(overrides: Partial<SparkWorkState> = {}): SparkWorkState {
  return { sorting: false, sortWorker: null, lodWorker: null, ...overrides };
}

describe("Spark worker activity", () => {
  it("counts a sort, a pending worker call or an exclusive LoD update as busy", () => {
    expect(sparkBusy(sparkState())).toBe(false);
    expect(sparkBusy(sparkState({ sortWorker: idleWorker(), lodWorker: idleWorker() }))).toBe(
      false,
    );
    expect(sparkBusy(sparkState({ sorting: true }))).toBe(true);
    expect(sparkBusy(sparkState({ sortWorker: { queue: null, messages: { 7: {} } } }))).toBe(true);
    expect(sparkBusy(sparkState({ lodWorker: { queue: [], messages: {} } }))).toBe(true);
  });
});

describe("disposing Spark once its workers are idle", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("releases at once when nothing is running", () => {
    const release = vi.fn();
    whenSparkIdle(sparkState(), release);
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("waits for a sort in flight, so its worker call is never rejected", () => {
    const spark = sparkState({ sorting: true });
    const release = vi.fn();
    whenSparkIdle(spark, release);
    vi.advanceTimersByTime(SPARK_IDLE_POLL_MS * 3);
    expect(release).not.toHaveBeenCalled();
    spark.sorting = false;
    vi.advanceTimersByTime(SPARK_IDLE_POLL_MS);
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("gives up waiting after the bound rather than keeping the renderer", () => {
    const release = vi.fn();
    whenSparkIdle(sparkState({ sorting: true }), release);
    vi.advanceTimersByTime(SPARK_IDLE_MAX_WAIT_MS - SPARK_IDLE_POLL_MS);
    expect(release).not.toHaveBeenCalled();
    vi.advanceTimersByTime(SPARK_IDLE_POLL_MS * 2);
    expect(release).toHaveBeenCalledTimes(1);
  });
});
