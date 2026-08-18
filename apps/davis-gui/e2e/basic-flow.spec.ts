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
  await expect(
    page.getByRole("combobox", { name: "rail variable beta_cost" }),
  ).toHaveValue("rail_cost");
  await expect(page.getByTestId("utility-bus").locator('input[value="beta_cost"]')).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "bus variable beta_cost" }),
  ).toHaveValue("bus_cost");
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
  await expect(page.locator(".model-status")).toHaveText("Draft");
  await expect(page.getByRole("combobox", { name: "car variable beta_time" })).toHaveValue("time_car");
  await expect(page.getByRole("combobox", { name: "car variable beta_cost" })).toHaveValue("cost_car");
  await expect(page.locator('option[value="car_time"]')).toHaveCount(0);

  await page.keyboard.press("ControlOrMeta+K");
  await expect(commandSearch).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(commandSearch).toBeHidden();
});

test("selects alternatives and explanatory variables from the active schema", async ({ page }) => {
  await page.goto("/");

  await page.keyboard.press("ControlOrMeta+K");
  const commandSearch = page.getByRole("combobox", { name: "Search DAVIS commands" });
  await commandSearch.fill("New MNL Model");
  await page.keyboard.press("Enter");

  await expect(page.getByText("Schema locked · 8 columns")).toBeVisible();
  await page.getByRole("button", { name: "Select alternatives" }).click();
  await page.getByTestId("alternative-option-walk").click();
  await expect(page.getByTestId("utility-walk")).toHaveCount(0);
  await expect(page.locator(".alternative-trigger")).toContainText("3 / 4");

  await page.getByTestId("alternative-option-bus").click();
  await expect(page.getByTestId("utility-bus")).toHaveCount(0);
  await expect(page.getByTestId("alternative-option-car")).toHaveAttribute("data-disabled", "");
  await expect(page.getByTestId("alternative-option-rail")).toHaveAttribute("data-disabled", "");

  await page.getByTestId("alternative-option-bus").click();
  await expect(page.getByTestId("utility-bus")).toBeVisible();
  await page.keyboard.press("Escape");

  const addRailVariable = page.getByRole("combobox", {
    name: "Add explanatory variable to RAIL",
  });
  await addRailVariable.selectOption("income");
  await expect(page.getByRole("combobox", { name: "rail variable beta_income" })).toHaveValue("income");
  await page.getByRole("button", { name: "Remove beta_income from rail" }).click();
  await expect(page.getByRole("combobox", { name: "rail variable beta_income" })).toHaveCount(0);

  const variableSelects = page.locator('.term-row select[aria-label*=" variable "]');
  const allowedTokyoColumns = [
    "car_time",
    "car_cost",
    "rail_time",
    "rail_cost",
    "bus_time",
    "bus_cost",
    "walk_time",
    "income",
  ];
  const selectedValues = await variableSelects.evaluateAll((selects) =>
    selects.map((select) => (select as HTMLSelectElement).value),
  );
  expect(selectedValues.every((value) => allowedTokyoColumns.includes(value))).toBe(true);
  const railVariableOptions = await page
    .getByRole("combobox", { name: "rail variable beta_time" })
    .locator("option")
    .evaluateAll((options) => options.map((option) => (option as HTMLOptionElement).value));
  expect(railVariableOptions).toEqual(allowedTokyoColumns);

  await page.getByRole("button", { name: "Dataset selector" }).click();
  await page.getByRole("menuitem").filter({ hasText: "Synthetic Commute" }).click();
  await expect(page.getByRole("combobox", { name: "car variable beta_time" })).toHaveValue("drive_minutes");
  await expect(page.getByRole("combobox", { name: "car variable beta_cost" })).toHaveValue("drive_fare");
  await expect(page.locator('option[value="car_time"]')).toHaveCount(0);
});
