import { expect, test, type Page } from "@playwright/test";

// The synthetic two-room fixture (packages/scene-contract/fixtures): outer walls
// at x = ±4 and z = ±2.5, a divider at x = 0 with one door, a table in the other
// room, spawn feet [2, 0, 0] facing -x. The walker's radius is 0.25 m.
const WALL_X = 4;
const WALL_Z = 2.5;
const SPAWN: [number, number, number] = [2, 0, 0];
const TABLE = { minX: -2.5, maxX: -1.5, minZ: -1, maxZ: 0 };
const RADIUS = 0.25;

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

/** Horizontal distance from the feet to the table's footprint (0 when on it). */
function tableClearance(p: Vec3): number {
  const dx = Math.max(TABLE.minX - p[0], 0, p[0] - TABLE.maxX);
  const dz = Math.max(TABLE.minZ - p[2], 0, p[2] - TABLE.maxZ);
  return Math.hypot(dx, dz);
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

  // The spawn faces -x: walking forward passes through the divider's door into the
  // other room and stops against what is ahead, the table or the far wall
  // (which one depends on how the door frame nudged the walker sideways).
  await page.keyboard.down("KeyW");
  await expect.poll(async () => (await feet(page))![0], { timeout: 20_000 }).toBeLessThan(-1);
  let previous = await settledFeet(page);
  await expect
    .poll(
      async () => {
        await page.waitForTimeout(1_000);
        const now = await settledFeet(page);
        const moved = horizontalDistance(now, previous);
        previous = now;
        return moved;
      },
      { timeout: 30_000, intervals: [0] },
    )
    .toBeLessThan(0.01);
  await page.keyboard.up("KeyW");
  const stopped = await settledFeet(page);
  expect(horizontalDistance(stopped, start)).toBeGreaterThan(3);
  expectInsideTheWalls(stopped);
  expect(tableClearance(stopped)).toBeGreaterThan(RADIUS - 0.05);

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

test("draws the textured model, not a blank canvas", async ({ page }) => {
  // Two animation frames after "ready": the frame loop has drawn the scene at least once.
  await page.evaluate(
    () =>
      new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))),
  );
  // The element screenshot includes whatever sits over the canvas, and the dev page's hint
  // text would count as variance on a blank frame, so it is hidden for the capture.
  await page.locator("#hint").evaluate((hint: HTMLElement) => {
    hint.style.display = "none";
  });
  const png = await page.locator("#stage canvas").screenshot();
  const stats = await page.evaluate(
    async (dataUrl) => {
      // Decoded in the browser, so no PNG library is needed; pixels are sampled on a grid.
      const image = new Image();
      image.src = dataUrl;
      await image.decode();
      const surface = document.createElement("canvas");
      surface.width = image.naturalWidth;
      surface.height = image.naturalHeight;
      const context = surface.getContext("2d", { willReadFrequently: true })!;
      context.drawImage(image, 0, 0);
      const { data, width, height } = context.getImageData(0, 0, surface.width, surface.height);
      let samples = 0;
      let sum = 0;
      let sumSquares = 0;
      let clear = 0;
      for (let y = 0; y < height; y += 5) {
        for (let x = 0; x < width; x += 5) {
          const i = (y * width + x) * 4;
          const r = data[i]!;
          const g = data[i + 1]!;
          const b = data[i + 2]!;
          const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
          samples += 1;
          sum += luminance;
          sumSquares += luminance * luminance;
          // The renderer clears to black, so a frame that drew nothing is all clear pixels.
          if (Math.max(r, g, b) <= 2) clear += 1;
        }
      }
      const mean = sum / samples;
      return {
        luminanceStdDev: Math.sqrt(sumSquares / samples - mean * mean),
        clearFraction: clear / samples,
      };
    },
    `data:image/png;base64,${png.toString("base64")}`,
  );
  // A textured frame varies across its pixels and has no clear ones; a blank frame has
  // no variance and only clear (black) pixels.
  expect(stats.luminanceStdDev).toBeGreaterThan(10);
  expect(stats.clearFraction).toBeLessThan(0.5);
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
