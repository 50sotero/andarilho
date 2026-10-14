import type { KeyValueStorage } from "@/lib/upload-resume";

export class MemoryStorage implements KeyValueStorage {
  readonly map = new Map<string, string>();
  get length() {
    return this.map.size;
  }
  key(index: number) {
    return [...this.map.keys()][index] ?? null;
  }
  getItem(key: string) {
    return this.map.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.map.set(key, value);
  }
  removeItem(key: string) {
    this.map.delete(key);
  }
}

export class ThrowingStorage implements KeyValueStorage {
  get length(): number {
    throw new Error("SecurityError");
  }
  key(): string | null {
    throw new Error("SecurityError");
  }
  getItem(): string | null {
    throw new Error("SecurityError");
  }
  setItem(): void {
    throw new Error("QuotaExceededError");
  }
  removeItem(): void {
    throw new Error("SecurityError");
  }
}
