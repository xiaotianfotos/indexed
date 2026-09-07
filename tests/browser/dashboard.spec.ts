import { test, expect } from "./fixtures.js";

test("first use, real document indexing, search and keyboard preview", async ({ page, app }) => {
  await page.goto(app.url);
  await expect(page.locator("#setup-title")).toHaveText("先设置检索模型");
  await expect(page.getByRole("textbox", { name: "检索素材" })).toBeDisabled();
  await expect(page.locator("#connection-pill")).toHaveText("本机服务在线");
  await page.locator("#setup-action").click();
  await expect(page.locator("#embedding-provider")).toHaveValue("remote");

  // Verify the default and the explicit experimental opt-in without starting a native model.
  await page.locator("#embedding-provider").selectOption("apple-native");
  await expect(page.locator("#apple-execution-mode")).toHaveValue("b");
  for (const mode of ["a", "c", "d"]) await expect(page.locator(`#apple-execution-mode option[value=${mode}]`)).toHaveJSProperty("disabled", true);
  await expect(page.locator("#apple-execution-mode option[value=e]")).toHaveCount(0);
  await page.locator("#apple-experiments").check();
  for (const mode of ["a", "c", "d"]) await expect(page.locator(`#apple-execution-mode option[value=${mode}]`)).toHaveJSProperty("disabled", false);
  await expect(page.getByRole("tab", { name: "入库测试" })).toBeHidden();
  await page.getByText("开发者工具", { exact: true }).click();
  await page.locator("#developer-tools").check();
  await expect(page.getByRole("tab", { name: "入库测试" })).toBeVisible();
  await page.locator("#developer-tools").uncheck();
  await expect(page.getByRole("tab", { name: "入库测试" })).toBeHidden();

  await page.locator("#embedding-provider").selectOption("remote");
  await page.locator("#embedding-model").fill("dashboard-fixture");
  await page.locator("#embedding-base-url").fill(app.embeddingUrl);
  await page.locator("#profile-panel details").evaluateAll(elements => {
    for (const element of elements) if (element instanceof HTMLDetailsElement && !element.hidden) element.open = true;
  });
  await page.locator("#embedding-dimension").fill("3");
  await page.locator("#embedding-space").fill("dashboard-fixture-space");
  await page.locator("#save-profile").click();
  await expect(page.locator("#profile-note")).toHaveText("已保存并启用");
  await page.locator("#view-toggle").click();
  await expect(page.locator("#setup-title")).toHaveText("添加第一个素材目录");
  await page.locator("#setup-action").click();
  await expect(page.getByRole("dialog", { name: "选择素材目录" })).toBeVisible();
  await expect(page.locator("#picker-input")).toBeFocused();
  await expect(page.locator(".app-shell")).toHaveAttribute("inert", "");
  await expect(page.locator("#picker-choose")).toBeEnabled();
  await page.locator("#picker-choose").focus();
  await page.keyboard.press("Tab");
  await expect(page.locator("#picker-close")).toBeFocused();
  await page.locator("#picker-input").fill(app.directory);
  await page.locator("#picker-input").press("Enter");
  await expect(page.locator("#picker-note")).toHaveText("0 个子目录");
  await expect(page.locator("#picker-input")).toHaveValue(app.directory);
  await page.locator("#picker-choose").click();
  await expect(page.locator("#picker")).toBeHidden();
  await expect(page.locator("#pick-directory")).toBeFocused();
  await page.locator("#view-toggle").click();
  await expect(page.getByRole("textbox", { name: "检索素材" })).toBeEnabled({ timeout: 20_000 });
  await page.getByRole("textbox", { name: "检索素材" }).fill("窗边的猫");
  await page.getByRole("button", { name: "搜索", exact: true }).click();
  const card = page.getByRole("button", { name: "打开 窗边的猫.txt" });
  await expect(card).toBeVisible();
  await expect(card.locator("img")).toHaveCount(0);
  await expect(page.locator("#asset-search-status")).toContainText("1 个结果");
  await card.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("dialog", { name: "窗边的猫.txt" })).toBeVisible();
  await expect(page.locator("#asset-modal-close")).toBeFocused();
  await expect(page.locator("#asset-document")).toHaveValue("一只猫坐在窗边，看着花园。\n");
  await page.keyboard.press("Escape");
  await expect(card).toBeFocused();
  await expect(page.locator(".app-shell")).not.toHaveAttribute("inert", "");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test("latest search wins and clearing the input cancels pending rendering", async ({ page, app }) => {
  // UI-level race fixture: a transport that deliberately completes after abort.
  await page.route("**/api/config", route => route.fulfill({ json: {
    activeProfile: "default", profiles: { default: { embedding: { provider: "remote", baseUrl: app.embeddingUrl } } },
  } }));
  await page.route("**/api/status", route => route.fulfill({ json: { storage: { collections: [{ count: 1 }] } } }));
  await page.addInitScript(() => {
    const original = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      if (String(input) !== "/api/assets/search") return original(input, init);
      const query: string = JSON.parse(String(init?.body)).query;
      await new Promise(resolve => setTimeout(resolve, query === "慢查询" ? 600 : 20));
      return new Response(JSON.stringify({ hits: [{ name: query, kind: "document", path: "/synthetic/document.txt" }], sources: { local: 1 } }),
        { status: 200, headers: { "content-type": "application/json" } });
    };
  });
  await page.goto(app.url);
  const input = page.getByRole("textbox", { name: "检索素材" });
  await expect(input).toBeEnabled();
  await input.fill("慢查询");
  await input.press("Enter");
  await expect(page.locator("#asset-search-status")).toHaveText("检索中");
  await input.fill("新查询");
  await input.press("Enter");
  await expect(page.getByRole("button", { name: "打开 新查询" })).toBeVisible();
  await page.waitForTimeout(700); // Both deterministic fixture responses have now completed.
  await expect(page.getByRole("button", { name: "打开 新查询" })).toBeVisible();
  await expect(page.getByRole("button", { name: "打开 慢查询" })).toHaveCount(0);
  await input.fill("慢查询");
  await input.press("Enter");
  await input.fill("");
  await page.waitForTimeout(700);
  await expect(page.locator("#asset-results")).toBeEmpty();
  await expect(page.locator("#asset-results")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#asset-search-status")).toBeEmpty();
});

test("model loading and failure can recover without reloading the page", async ({ page, app }) => {
  let backendState = "starting";
  let offline = false;
  await page.route("**/api/config", route => route.fulfill(offline
    ? { status: 503, json: { error: "合成服务离线" } }
    : { json: {
      activeProfile: "default",
      profiles: { default: { embedding: { provider: "apple-native", native: { executionMode: "b" } } } },
    } }));
  await page.route("**/api/embedding-backend", route => route.fulfill({ json: { state: backendState, lastError: "合成模型加载失败" } }));
  await page.route("**/api/status", route => route.fulfill({ json: { storage: { collections: [{ count: 1 }] } } }));
  await page.goto(app.url);
  const input = page.getByRole("textbox", { name: "检索素材" });
  await expect(page.locator("#setup-title")).toHaveText("模型正在准备");
  await expect(input).toBeDisabled();
  backendState = "failed";
  await page.locator("#connection-pill").click();
  await expect(page.locator("#setup-title")).toHaveText("连接检查未通过");
  await expect(page.locator("#setup-detail")).toHaveText("合成模型加载失败");
  backendState = "ready";
  await page.locator("#connection-pill").click();
  await expect(input).toBeEnabled();
  await expect(page.locator("#setup-guide")).toBeHidden();
  offline = true;
  await page.locator("#connection-pill").click();
  await expect(page.locator("#setup-title")).toHaveText("无法连接本机服务");
  await expect(input).toBeDisabled();
  offline = false;
  await page.locator("#setup-action").click();
  await expect(input).toBeEnabled();
});
