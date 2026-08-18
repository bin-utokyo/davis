import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";

const assertNoHorizontalOverflow = async (page: Page) => {
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
};

test("captures the dark alternative selector at 1440 by 900", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByText("Schema locked · 8 columns")).toBeVisible();
  await page.getByRole("button", { name: "Select alternatives" }).click();
  await expect(page.getByRole("menuitemcheckbox", { name: "WALK" })).toBeVisible();
  await assertNoHorizontalOverflow(page);
  await page.screenshot({ path: "docs/screenshots/davis-gui-1440x900.png" });
});

test("captures a light dataset-specific specification at 1280 by 800", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  await page.getByRole("button", { name: "Switch to light mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "Dataset selector" }).click();
  await page.getByRole("menuitem").filter({ hasText: "Teaching Sample" }).click();
  await expect(page.getByRole("combobox", { name: "car variable beta_time" })).toHaveValue("time_car");
  await assertNoHorizontalOverflow(page);
  await page.screenshot({ path: "docs/screenshots/davis-gui-1280x800.png" });
});
