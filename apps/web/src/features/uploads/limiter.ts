/** A tiny counting semaphore used to cap concurrent storage transfers across files. */
export interface Limiter {
  run<T>(task: () => Promise<T>): Promise<T>;
  readonly active: number;
}

export function createLimiter(max: number): Limiter {
  let active = 0;
  const waiting: (() => void)[] = [];

  const acquire = () =>
    new Promise<void>((resolve) => {
      if (active < max) {
        active += 1;
        resolve();
      } else {
        waiting.push(() => {
          active += 1;
          resolve();
        });
      }
    });

  const release = () => {
    active -= 1;
    const next = waiting.shift();
    if (next) next();
  };

  return {
    async run<T>(task: () => Promise<T>): Promise<T> {
      await acquire();
      try {
        return await task();
      } finally {
        release();
      }
    },
    get active() {
      return active;
    },
  };
}
