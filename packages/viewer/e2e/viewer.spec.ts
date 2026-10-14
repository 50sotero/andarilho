import { expect, test, type Page } from "@playwright/test";

// The synthetic two-room fixture (packages/scene-contract/fixtures): outer walls
// at x = ±4 and z = ±2.5, a divider at x = 0 with one door, spawn feet [2, 0, 0]
// facing -x.
const WALL_X = 4;
const WALL_Z = 2.5;
const SPAWN: [number, number, number] = [2, 0, 0];

type Vec3 = [number, number, number];

/** The dev page's "feet x, y, z" readout (refreshed every 250 ms). */
async function feet(page: Page): Promise<Vec3 | null> {
  const text = (await page.locator("#status + span").textContent()) ?? "";
  const match = /^feet (-?[\d.]+), (-?[\d.]+), (-?[\d.]+)$/.exec(text.trim());
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null;
}

async function settledFeet(page: Page): Promise<Vec3> {
  await expect.poll(() => feet(page)).not.toBeNull();
  return (await feet(page))!;
}

function horizontalDistance(a: Vec3, b: Vec3): number {
  return Math.hypot(a[0] - b[0], a[2] - b[2]);
}

function expectInsideTheWalls(p: Vec3): void {
  expect(Math.abs(p[0])).toBeLessThan(WALL_X);
  expect(Math.abs(p[2])).toBeLessThan(WALL_Z);
  expect(Math.abs(p[1])).toBeLessThan(0.25); // on the floor, not falling
}

async function hold(page: Page, key: string, ms: number): Promise<void> {
  await page.keyboard.down(key);
  await page.waitForTimeout(ms);
  await page.keyboard.up(key);
}

/**
 * Releases pointer lock, which leaves walk mode on desktop. A user presses
 * Escape, but the browser handles that key in its own UI, which headless
 * Chromium lacks, so the page releases the lock the same way the browser does.
 */
async function releasePointer(page: Page): Promise<void> {
  await page.evaluate(() => document.exitPointerLock());
  await expect(page.locator("#status")).toContainText("ready (walkable");
}

let pageErrors: string[] = [];

test.beforeEach(async ({ page }) => {
  pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/");
  await expect(page.locator("#status")).toContainText("ready (walkable");
});

test.afterEach(() => {
  expect(pageErrors).toEqual([]);
});

test("loads the synthetic fixture as a walkable scene at its spawn", async ({ page }) => {
  await expect(page.locator("#walk")).toBeEnabled();
  await expect(page.locator("#exit")).toBeDisabled();
  await expect(page.locator("#stage canvas")).toBeVisible();
  const start = await settledFeet(page);
  expect(horizontalDistance(start, SPAWN)).toBeLessThan(0.05);
  expectInsideTheWalls(start);
});

test("walks with the keyboard, collides with the walls and resets to the spawn", async ({
  page,
}) => {
  await page.locator("#walk").click();
  await expect(page.locator("#status")).toContainText("walking (");
  await expect(page.locator("#status")).toContainText("pointer locked");
  await expect(page.locator("#exit")).toBeEnabled();
  const start = await settledFeet(page);

  // The spawn faces -x: walking forward passes through the divider's door into
  // the other room and stops at its far wall.
  await page.keyboard.down("KeyW");
  await expect.poll(async () => (await feet(page))![0], { timeout: 20_000 }).toBeLessThan(-3);
  await page.waitForTimeout(1_500);
  await page.keyboard.up("KeyW");
  const stopped = await settledFeet(page);
  expect(horizontalDistance(stopped, start)).toBeGreaterThan(5);
  expect(stopped[0]).toBeGreaterThan(-WALL_X);
  expectInsideTheWalls(stopped);

  await hold(page, "ArrowLeft", 2_000);
  expectInsideTheWalls(await settledFeet(page));

  // While the pointer is locked the header cannot be clicked.
  await releasePointer(page);
  await page.locator("#reset").click();
  await expect.poll(async () => horizontalDistance((await feet(page))!, SPAWN)).toBeLessThan(0.05);
});

test("ignores movement keys after leaving walk mode", async ({ page }) => {
  await page.locator("#walk").click();
  await expect(page.locator("#status")).toContainText("walking");
  await releasePointer(page);
  const before = await settledFeet(page);
  await hold(page, "KeyW", 1_500);
  await page.waitForTimeout(500);
  expect(horizontalDistance((await settledFeet(page))!, before)).toBeLessThan(0.01);
});

test("builds walkable collision from the layout in the browser", async ({ page }) => {
  await page.locator("#collision").selectOption("layout");
  await expect(page.locator("#status")).toContainText("ready (walkable");
  await page.locator("#walk").click();
  await expect(page.locator("#status")).toContainText("walking");
  const start = await settledFeet(page);
  await hold(page, "KeyW", 5_000);
  const end = await settledFeet(page);
  expect(horizontalDistance(end, start)).toBeGreaterThan(0.5);
  expectInsideTheWalls(end);
});
