import { expect, test } from "@playwright/test";

test("creates, improves, estimates, and restores an MNL model", async ({ page }) => {
  await page.goto("/");

  await page.keyboard.press("ControlOrMeta+K");
  const commandSearch = page.getByRole("combobox", { name: "Search DAVIS commands" });
  await expect(commandSearch).toBeVisible();
  await commandSearch.fill("New MNL Model");
  await page.keyboard.press("Enter");

  await expect(page.locator(".model-status")).toHaveText("Draft");
  await expect(page.getByTestId("utility-car").locator('input[value="ASC_car"]')).toBeVisible();
  await expect(page.getByTestId("utility-car").locator('input[value="beta_cost"]')).toBeVisible();
  await expect(page.getByTestId("utility-rail").locator('input[value="beta_cost"]')).toHaveCount(0);
  await expect(page.getByTestId("utility-bus").locator('input[value="beta_cost"]')).toHaveCount(0);

  const costSuggestion = page.locator(".suggestion-card").filter({ hasText: "Cost appears only in Car." });
  await costSuggestion.getByRole("button", { name: /Apply suggestion/ }).click();

  await expect(page.locator(".model-status")).toHaveText("Modified");
  await expect(page.getByTestId("utility-rail").locator('input[value="beta_cost"]')).toBeVisible();
  await expect(page.getByTestId("utility-rail").locator('input[value="rail_cost"]')).toBeVisible();
  await expect(page.getByTestId("utility-bus").locator('input[value="beta_cost"]')).toBeVisible();
  await expect(page.getByTestId("utility-bus").locator('input[value="bus_cost"]')).toBeVisible();
  await expect(
    page.locator(".suggestion-card").filter({ hasText: "Cost appears only in Car." }),
  ).toHaveCount(0);

  await page.getByRole("button", { name: "Run Estimation" }).click();
  await expect(page.getByRole("button", { name: "Estimating..." })).toBeVisible();
  await expect(page.getByTestId("result-status")).toContainText("Converged");

  await page.getByRole("tab", { name: "Coefficients" }).click();
  await expect(page.getByLabel("Coefficient confidence intervals")).toBeVisible();

  await page.getByRole("button", { name: /MNL-03/ }).click();
  await expect(page.locator(".model-status")).toHaveText("Saved");
  await expect(page.getByTestId("utility-rail").locator('input[value="ASC_rail"]')).toBeVisible();
  await expect(page.getByTestId("utility-bus").locator('input[value="ASC_bus"]')).toBeVisible();
  await expect(page.getByLabel("Coefficient confidence intervals")).toBeVisible();
  await expect(page.getByText("ρ² 0.24", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "Table" }).click();
  await expect(page.getByText("-3912.4", { exact: true })).toBeVisible();
});

test("supports command-palette keyboard control and dataset switching", async ({ page }) => {
  await page.goto("/");

  await page.keyboard.press("ControlOrMeta+K");
  const commandSearch = page.getByRole("combobox", { name: "Search DAVIS commands" });
  await commandSearch.fill("Choose Dataset");
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowUp");
  await page.keyboard.press("Enter");

  const teachingSample = page.getByRole("menuitem").filter({ hasText: "Teaching Sample" });
  await expect(teachingSample).toBeVisible();
  await teachingSample.click();
  await expect(page.getByRole("button", { name: "Dataset selector" })).toContainText("Teaching Sample");

  await page.keyboard.press("ControlOrMeta+K");
  await expect(commandSearch).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(commandSearch).toBeHidden();
});
