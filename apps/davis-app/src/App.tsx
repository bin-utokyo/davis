import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useEffect, useRef, useState } from "react";
import { bindingColumns, ColumnBinding, FormDefinition, FormInput, FormJoin, FormSource, JsonSchema, SchemaFormEditor, SchemaFormErrorBoundary } from "./SchemaFormEditor";
import { localizedText, localizeTree, useI18n } from "./i18n";

type ColumnProfile = { name: string; inferred_type: string; null_count: number; unique_sample: number; warnings: string[] };
type CsvProfile = { path: string; encoding: string; delimiter: string; rows_sampled: number; truncated: boolean; columns: ColumnProfile[] };
type Validation = { valid: boolean; plan: string; component: { id: string; version: string; manifest: string } };
type ArtifactProfile = "table" | "metrics" | "parameters" | "predictions" | "figure" | "diagnostics" | "report";
type Artifact = { path: string; media_type: string; profile?: ArtifactProfile; size?: number };
type CompletedRun = {
  run_directory: string; request: { run_id: string; operation: string; component: { id: string; version: string; kind?: string } };
  result: { status: string; artifacts: Record<string, Artifact>; extensions: Record<string, Artifact> };
};
type RunHistoryEntry = { run: CompletedRun; plan_name: string; label?: string; tags: string[]; modified_at_unix_ms: number };
type RunHistoryResponse = { runs: RunHistoryEntry[]; warnings: string[] };
type RunComparison = { entry: RunHistoryEntry; metrics: Record<string, unknown>; parameters: Record<string, Record<string, string>> };
type ComponentEditor = {
  manifest: { id: string; name: string; version: string; kind: string; operations: string[]; inputs: Array<{ name: string; required: boolean }>; outputs: { artifacts: Record<string, { profile?: ArtifactProfile; media_types: string[]; required: boolean }> } };
  config_schema: JsonSchema; ui_schema: FormDefinition; ui_extensions: Record<string, { api_version: string; html: string }>;
};
type PlanInput = {
  kind: string; path?: string; read?: unknown; dataset_id?: string; file_id?: string; revision?: string; processor?: { id: string; version: string }; sources?: Record<string, PlanInput>; base?: string;
  joins?: Array<{ source: string; left_on: string; right_on: string; relationship?: FormJoin["relationship"]; how?: FormJoin["how"]; allow_unmatched?: boolean }>;
  columns?: Record<string, ColumnBinding>;
};
type EditablePlan = {
  yaml: string; resolved_sources: Record<string, string>;
  plan: { name: string; component: { id: string; version: string; operation: string }; inputs: Record<string, PlanInput>; config: Record<string, unknown>; run?: { label?: string; tags?: string[]; [key: string]: unknown } };
  editor: ComponentEditor;
};
type ArtifactPreview = { name: string; media_type: string; content: unknown };
type ResultDefinition = { artifact: string; title: string; widget: "key-value" | "table"; profile?: ArtifactProfile };
type CatalogFile = { dataset_id: string; file_id: string; path: string; title: string; size: number; columns: string[] };
type DownloadedCatalogFile = { dataset_id: string; file_id: string; path: string };

export default function App() {
  const { locale, setLocale, t } = useI18n();
  const [repository, setRepository] = useState(""); const [planPath, setPlanPath] = useState("");
  const [planName, setPlanName] = useState("analysis-plan"); const [runLabel, setRunLabel] = useState("");
  const [editor, setEditor] = useState<ComponentEditor>(); const [editorOptions, setEditorOptions] = useState<ComponentEditor[]>([]);
  const [inputs, setInputs] = useState<Record<string, FormInput | undefined>>({}); const [config, setConfig] = useState<Record<string, unknown>>({});
  const [preservedRun, setPreservedRun] = useState<Record<string, unknown>>({}); const [yamlPreview, setYamlPreview] = useState("");
  const [codeMode, setCodeMode] = useState(false); const [validation, setValidation] = useState<Validation>();
  const [completed, setCompleted] = useState<CompletedRun>(); const [completedEditor, setCompletedEditor] = useState<ComponentEditor>(); const [artifactPreviews, setArtifactPreviews] = useState<Record<string, ArtifactPreview>>({});
  const [runHistory, setRunHistory] = useState<RunHistoryEntry[]>([]); const [historyWarnings, setHistoryWarnings] = useState<string[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]); const [comparison, setComparison] = useState<RunComparison[]>([]);
  const [catalogTarget, setCatalogTarget] = useState<string>(); const [catalogFiles, setCatalogFiles] = useState<CatalogFile[]>([]); const [catalogSearch, setCatalogSearch] = useState("");
  const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const resultRef = useRef<HTMLElement>(null);

  useEffect(() => { if (completed) requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })); }, [completed]);
  async function perform(action: () => Promise<void>) { setBusy(true); setError(""); try { await action(); } catch (reason) { setError(String(reason)); } finally { setBusy(false); } }
  async function refreshHistory(workspace = repository) {
    const history = await invoke<RunHistoryResponse>("list_project_runs", { repository: workspace });
    setRunHistory(history.runs); setHistoryWarnings(history.warnings);
    setSelectedRunIds((current) => current.filter((id) => history.runs.some((entry) => entry.run.request.run_id === id)));
  }

  async function chooseRepository() {
    const selected = await open({ directory: true, multiple: false }); if (typeof selected !== "string") return;
    setRepository(selected);
    await perform(async () => {
      const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository: selected }); const definition = definitions[0];
      setEditorOptions(definitions);
      if (definition) selectDefinition(definition); else { setEditor(undefined); setCompleted(undefined); setArtifactPreviews({}); }
      await refreshHistory(selected);
    });
  }
  async function reloadComponents() {
    if (!repository) return;
    await perform(async () => {
      const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository });
      setEditorOptions(definitions);
      if (definitions.length) selectDefinition(definitions[0]);
    });
  }
  function selectDefinition(definition: ComponentEditor) {
    setEditor(definition); setPlanName(defaultPlanName(definition)); setRunLabel(""); setInputs({});
    setConfig(structuredClone(definition.ui_schema.defaults ?? {})); setPreservedRun({}); setPlanPath(""); setYamlPreview("");
    setCodeMode(false); setValidation(undefined); setCompleted(undefined); setCompletedEditor(undefined); setArtifactPreviews({});
  }
  function selectEditor(identity: string) { const definition = editorOptions.find((item) => `${item.manifest.id}@${item.manifest.version}` === identity); if (definition) selectDefinition(definition); }
  async function newPlan() {
    await perform(async () => {
      let definition = editor;
      if (!definition || !isComposedEditor(definition)) { const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository }); setEditorOptions(definitions); definition = definitions[0]; }
      if (!definition) throw new Error(t("利用できるcomponentがありません．先にDavis CLIで公式componentをインストールしてください．")); selectDefinition(definition);
    });
  }

  async function addSources(slot: string) {
    const selected = await open({ multiple: true, filters: [{ name: "CSV", extensions: ["csv", "tsv"] }] });
    if (!Array.isArray(selected) || !selected.length) return;
    await perform(async () => {
      const inspected = await Promise.all(selected.map(async (path) => ({ path, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) })));
      setInputs((current) => {
        const existing = current[slot]; const additions: FormSource[] = [];
        for (const item of inspected) additions.push({ id: uniqueSourceId(item.path, [...(existing?.sources ?? []), ...additions]), path: item.path, serializedPath: item.path, origin: { kind: "local" }, profile: item.profile });
        const sources = [...(existing?.sources ?? []), ...additions];
        return { ...current, [slot]: { sources, base: existing?.base ?? sources[0].id, joins: existing?.joins ?? {}, columns: existing?.columns, processor: existing?.processor, forceBinding: existing?.forceBinding } };
      });
    });
  }

  async function openCatalog(slot: string) {
    await perform(async () => { setCatalogFiles(await invoke<CatalogFile[]>("catalog_files", { locale })); setCatalogSearch(""); setCatalogTarget(slot); });
  }
  async function selectCatalogFile(file: CatalogFile) {
    if (!catalogTarget) return;
    await perform(async () => {
      const downloaded = await invoke<DownloadedCatalogFile>("download_catalog_file", { datasetId: file.dataset_id, fileId: file.file_id, locale });
      const profile = await invoke<CsvProfile>("inspect_csv_file", { path: downloaded.path });
      const slot = catalogTarget;
      setInputs((current) => {
        const existing = current[slot];
        const source: FormSource = { id: uniqueSourceId(file.path, existing?.sources ?? []), path: downloaded.path, serializedPath: file.path, profile, origin: { kind: "catalog", dataset_id: file.dataset_id, file_id: file.file_id } };
        const sources = [...(existing?.sources ?? []), source];
        return { ...current, [slot]: { sources, base: existing?.base ?? source.id, joins: existing?.joins ?? {}, columns: existing?.columns, processor: existing?.processor, forceBinding: existing?.forceBinding } };
      });
      setCatalogTarget(undefined);
    });
  }

  function buildPlan(outputPath?: string): Record<string, unknown> {
    if (!editor || !isComposedEditor(editor)) throw new Error(t("このcomponentはdavis.ui/v1に対応していません．"));
    const missing = editor.manifest.inputs.filter((item) => item.required).find((item) => !inputs[item.name]);
    if (missing) throw new Error(t("{id}の入力データを選択してください．").replace("{id}", missing.name));
    const absolutePaths = Boolean(outputPath && planPath && outputPath !== planPath); const name = planName.trim() || "analysis-plan";
    return {
      api_version: "davis.analysis/v1alpha1", name,
      component: { id: editor.manifest.id, version: editor.manifest.version, operation: editor.manifest.operations.includes("estimate") ? "estimate" : editor.manifest.operations[0] },
      inputs: Object.fromEntries(Object.entries(inputs).filter(([, input]) => input).map(([slot, input]) => [slot, buildInputBinding(slot, input!, absolutePaths)])),
      config, run: buildRunMetadata(["gui", editor.manifest.id.replace("/", "-")]),
    };
  }
  function buildInputBinding(slot: string, input: FormInput, absolutePaths: boolean): Record<string, unknown> {
    if (input.sources.length === 1 && !input.forceBinding) return serializeSource(input.sources[0], absolutePaths);
    const joins = input.sources.filter((source) => source.id !== input.base).map((source) => { const join = input.joins[source.id]; if (!join?.leftOn || !join.rightOn) throw new Error(t("{slot}の{source}について結合キーを選択してください．").replace("{slot}", slot).replace("{source}", source.id)); return { source: source.id, left_on: join.leftOn, right_on: join.rightOn, relationship: join.relationship, how: join.how, allow_unmatched: join.allowUnmatched }; });
    const preparation = editor?.ui_schema.inputs?.[slot]?.preparation;
    return { kind: "table_binding", processor: input.processor ?? (preparation ? { id: preparation.component, version: preparation.version } : { id: "davis/csv-transform", version: "0.4.0" }),
      sources: Object.fromEntries(input.sources.map((source) => [source.id, serializeSource(source, absolutePaths)])),
      base: input.base, joins, columns: Object.fromEntries(bindingColumns(input).map(({ alias, source, column }) => [alias, { source, column }])) };
  }
  function buildRunMetadata(defaultTags: string[]) { const metadata = { ...preservedRun }; delete metadata.label; if (runLabel.trim()) metadata.label = runLabel.trim(); if (!Array.isArray(metadata.tags)) metadata.tags = defaultTags; return metadata; }

  async function previewPlan() { await perform(async () => setYamlPreview(await invoke<string>("render_analysis_plan", { plan: buildPlan() }))); }
  async function saveDraft(execute: boolean, overwrite = false) {
    await perform(async () => {
      const target = overwrite ? planPath : await save({ defaultPath: repository ? `${repository}/model.yaml` : "model.yaml", filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] });
      if (!target) return; const plan = buildPlan(target); const saved = await invoke<string>("save_analysis_plan", { repository, path: target, plan });
      setPlanPath(saved); setYamlPreview(await invoke<string>("render_analysis_plan", { plan })); setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: saved })); setCompleted(undefined); setCompletedEditor(undefined);
      if (execute) await showCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: saved }));
    });
  }
  async function openPlanForEditing() {
    if (!repository) return setError(t("先にWorkspaceを選択してください．"));
    const selected = await open({ multiple: false, filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] }); if (typeof selected !== "string") return;
    await perform(async () => {
      const loaded = await invoke<EditablePlan>("load_analysis_plan_for_editing", { repository, path: selected });
      setEditorOptions((current) => current.some((item) => item.manifest.id === loaded.editor.manifest.id && item.manifest.version === loaded.editor.manifest.version) ? current : [...current, loaded.editor]);
      if (isComposedEditor(loaded.editor)) await hydratePlan(loaded, selected); else { setEditor(loaded.editor); setPlanPath(selected); setRunLabel(loaded.plan.run?.label ?? ""); setYamlPreview(loaded.yaml); setCodeMode(true); setCompleted(undefined); setCompletedEditor(undefined); setArtifactPreviews({}); setValidation(undefined); }
    });
  }
  async function hydratePlan(loaded: EditablePlan, path: string) {
    const loadedInputs: Record<string, FormInput | undefined> = {}; const issues: string[] = [];
    setEditor(loaded.editor); setPlanName(loaded.plan.name); setRunLabel(loaded.plan.run?.label ?? ""); setConfig(structuredClone(loaded.plan.config)); setPreservedRun(loaded.plan.run ?? {});
    setPlanPath(path); setYamlPreview(loaded.yaml); setCodeMode(false); setValidation(undefined); setCompleted(undefined); setCompletedEditor(undefined); setArtifactPreviews({});
    for (const declaration of loaded.editor.manifest.inputs) { const input = loaded.plan.inputs[declaration.name]; if (!input) continue; try { loadedInputs[declaration.name] = await hydrateInputBinding(declaration.name, input, loaded.resolved_sources); } catch (reason) { issues.push(`${t("読み込めませんでした")}: ${declaration.name}: ${String(reason)}`); } }
    setInputs(loadedInputs); if (issues.length) setError(issues.join("\n"));
  }
  async function hydrateInputBinding(slot: string, input: PlanInput, resolved: Record<string, string>): Promise<FormInput> {
    if (input.kind === "local" || input.kind === "catalog") { const source = await hydrateSource(slot, input, resolved, t); return { sources: [source], base: source.id, joins: {} }; }
    if (input.kind !== "table_binding" || !input.sources || !input.base || !input.columns) throw new Error(t("GUIで扱えない入力形式です．"));
    const sources: FormSource[] = await Promise.all(Object.entries(input.sources).map(async ([id, source]) => ({ ...(await hydrateSource(`${slot}/${id}`, source, resolved, t)), id })));
    const joins = Object.fromEntries((input.joins ?? []).map((join) => [join.source, { leftOn: join.left_on, rightOn: join.right_on, relationship: join.relationship ?? "many_to_one", how: join.how ?? "left", allowUnmatched: join.allow_unmatched ?? false } satisfies FormJoin]));
    return { sources, base: input.base, joins, columns: input.columns, processor: input.processor, forceBinding: true };
  }

  async function saveCodePlan(execute: boolean) { await perform(async () => { if (!planPath) throw new Error(t("保存先がありません．既存Planを開き直してください．")); await invoke<string>("save_analysis_plan_yaml", { repository, path: planPath, yaml: yamlPreview }); setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: planPath })); if (execute) await showCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: planPath })); }); }
  async function showCompleted(run: CompletedRun, resultEditor = editor) {
    setCompleted(run); setCompletedEditor(resultEditor); const definitions = resultDefinitions(resultEditor, run, locale);
    const previews = await Promise.all(definitions.map(async (definition) => { try { return await invoke<ArtifactPreview>("preview_run_artifact", { repository, runId: run.request.run_id, artifact: definition.artifact }); } catch { return undefined; } }));
    setArtifactPreviews(Object.fromEntries(previews.filter(Boolean).map((preview) => [preview!.name, preview!])));
    await refreshHistory();
  }
  async function openHistoricalRun(entry: RunHistoryEntry) { await perform(async () => {
    const identity = entry.run.request.component; let resultEditor = editorOptions.find((item) => item.manifest.id === identity.id && item.manifest.version === identity.version);
    if (!resultEditor) { try { resultEditor = await invoke<ComponentEditor>("component_editor_definition", { repository, componentId: identity.id, version: identity.version }); } catch { resultEditor = undefined; } }
    await showCompleted(await invoke<CompletedRun>("load_project_run", { repository, runId: entry.run.request.run_id }), resultEditor);
  }); }
  function toggleRun(runId: string) { setSelectedRunIds((current) => current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId]); setComparison([]); }
  async function compareSelectedRuns() { await perform(async () => {
    const selected = selectedRunIds.map((id) => runHistory.find((entry) => entry.run.request.run_id === id)).filter((entry): entry is RunHistoryEntry => Boolean(entry));
    const compared = await Promise.all(selected.map(async (entry): Promise<RunComparison> => {
      const metrics: Record<string, unknown> = {}; const parameters: Record<string, Record<string, string>> = {};
      for (const [artifact, descriptor] of [...Object.entries(entry.run.result.artifacts), ...Object.entries(entry.run.result.extensions)]) {
        if (!descriptor.profile || !["metrics", "parameters"].includes(descriptor.profile)) continue;
        try {
          const preview = await invoke<ArtifactPreview>("preview_run_artifact", { repository, runId: entry.run.request.run_id, artifact });
          if (descriptor.profile === "metrics") Object.assign(metrics, comparisonMetrics(preview.content, artifact));
          if (descriptor.profile === "parameters") Object.assign(parameters, comparisonParameters(preview.content));
        } catch { /* Unsupported artifact formats stay visible in the run detail. */ }
      }
      return { entry, metrics, parameters };
    }));
    setComparison(compared);
  }); }
  async function openRunDirectory() { if (completed) await perform(async () => invoke("open_run_directory", { repository, runId: completed.request.run_id })); }

  const editorReady = repository.length > 0 && (editor?.manifest.inputs.filter((item) => item.required).every((item) => Boolean(inputs[item.name]?.sources.length)) ?? false);
  const localizedEditor = editor ? { ...editor, config_schema: localizeTree(editor.config_schema, locale), ui_schema: localizeTree(editor.ui_schema, locale) } : undefined;
  return <main><header><div className="header-top"><p className="eyebrow">DAVIS MODEL</p><label className="language-select"><span>{locale === "ja" ? "言語" : "Language"}</span><select value={locale} onChange={(event) => setLocale(event.target.value as "ja" | "en")}><option value="ja">日本語</option><option value="en">English</option></select></label></div><h1>{t("ローカルデータから推定まで")}</h1><p className="lead">{t("ComponentManifestに従い，入力結合とモデル設定を同じAnalysisPlanとして編集します．")}</p></header>
    {error && <div className="error sticky-error">{error}</div>}
    <section><SectionHeading number="1" title="Project workspace" description={t("model.yamlとdavis-runsを置く作業folderです．Davis repositoryのcloneは不要です．")} /><PathField value={repository} placeholder="Project folder" onChange={(value) => { setRepository(value); setEditor(undefined); }} onChoose={chooseRepository} /></section>
    <section><div className="heading-with-actions"><SectionHeading number="2" title="Analysis plan editor" description={t("すべてのcomponentを同じdavis.ui/v1 rendererで編集します．")} /><div className="top-actions"><button className="secondary" onClick={newPlan}>{t("新規Plan")}</button><button className="secondary" disabled={!repository} onClick={openPlanForEditing}>{t("既存Planを開く")}</button></div></div>
      {repository && !editorOptions.length && <div className="notice component-install-notice"><strong>{t("componentがまだインストールされていません．")}</strong><p>{t("Project workspaceは正しく選択されています．次のcommandをターミナルへ貼り付けて公式componentをインストールしてください．")}</p><pre><code>{`davis install component mnl
davis install component nl
davis install component rl
davis install component csv-transform`}</code></pre><button className="secondary" disabled={busy} onClick={reloadComponents}>{t("Componentを再読み込み")}</button></div>}
      <div className="field-grid compact-grid"><label><span>Plan name</span><input value={planName} onChange={(event) => setPlanName(event.target.value)} /></label><label><span>Run name (folder prefix)</span><input value={runLabel} disabled={codeMode} placeholder={t("空欄ならPlan name")} onChange={(event) => setRunLabel(event.target.value)} /></label><label><span>Component Manifest</span><select value={editor ? `${editor.manifest.id}@${editor.manifest.version}` : ""} disabled={!editorOptions.length || codeMode} onChange={(event) => selectEditor(event.target.value)}>{!editor && <option value="">{t("Workspaceを選択してください")}</option>}{editorOptions.map((item) => <option key={`${item.manifest.id}@${item.manifest.version}`} value={`${item.manifest.id}@${item.manifest.version}`}>{componentDisplayName(item, locale)} ({item.manifest.id} {item.manifest.version})</option>)}</select></label></div>
      {codeMode && <div className="code-mode"><div className="notice">{t("このcomponentにはdavis.ui/v1の画面定義がありません．内容を失わないYAML modeで開いています．")}</div><textarea className="yaml-preview editable" value={yamlPreview} onChange={(event) => setYamlPreview(event.target.value)} aria-label="model.yaml code editor" /><div className="actions"><button className="secondary" disabled={busy} onClick={() => saveCodePlan(false)}>{t("上書き保存・検証")}</button><button disabled={busy} onClick={() => saveCodePlan(true)}>{t("上書きして実行")}</button></div></div>}
      {!codeMode && localizedEditor && isComposedEditor(localizedEditor) && <><SchemaFormErrorBoundary resetKey={`${localizedEditor.manifest.id}@${localizedEditor.manifest.version}`}><SchemaFormEditor definition={localizedEditor} inputs={inputs} config={config} onAddSources={addSources} onAddCatalog={openCatalog} onInputChange={(slot, input) => setInputs((current) => ({ ...current, [slot]: input }))} onConfigChange={setConfig} /></SchemaFormErrorBoundary>
        <div className="actions editor-actions"><button className="secondary" disabled={!editorReady || busy} onClick={previewPlan}>{t("YAMLを確認")}</button><button className="secondary" disabled={!editorReady || busy} onClick={() => saveDraft(false)}>{t("別名で保存")}</button><button className="secondary" disabled={!editorReady || !planPath || busy} onClick={() => saveDraft(false, true)}>{t("上書き保存")}</button><button disabled={!editorReady || busy} onClick={() => saveDraft(true, Boolean(planPath))}>{t(planPath ? "上書きして推定" : "保存して推定")}</button></div>
        {validation && <div className="success">{validation.component.id} {validation.component.version} {t("として保存・検証しました．")}</div>}{planPath && <div className="plan-path">{planPath}</div>}{yamlPreview && <textarea className="yaml-preview" readOnly value={yamlPreview} aria-label={t("生成されたmodel.yaml")} />}</>}
    </section>
    {repository && <RunHistoryPanel entries={runHistory} warnings={historyWarnings} selected={selectedRunIds} comparison={comparison} busy={busy} locale={locale} onRefresh={() => perform(() => refreshHistory())} onToggle={toggleRun} onOpen={openHistoricalRun} onCompare={compareSelectedRuns} />}
    {completed && <section ref={resultRef}><SectionHeading number="4" title="Run result" description={completed.request.run_id} /><div className="result-views">{resultDefinitions(completedEditor, completed, locale).map((definition) => { const preview = artifactPreviews[definition.artifact]; return preview ? <ResultPreview key={definition.artifact} definition={definition} preview={preview} /> : null; })}</div><div className="run-directory-row"><div className="run-directory">{completed.run_directory}</div><button className="secondary" disabled={busy} onClick={openRunDirectory}>{t("結果フォルダを開く")}</button></div><div className="artifacts">{[...Object.entries(completed.result.artifacts), ...Object.entries(completed.result.extensions)].map(([name, artifact]) => <article key={name}><strong>{name}</strong><span>{artifact.path}</span><small>{artifact.profile ? `${artifact.profile} · ` : ""}{artifact.media_type}{artifact.size ? ` · ${artifact.size} bytes` : ""}</small></article>)}</div></section>}
    {catalogTarget && <div className="modal-backdrop" onMouseDown={() => setCatalogTarget(undefined)}><div className="catalog-dialog" onMouseDown={(event) => event.stopPropagation()}><div className="catalog-heading"><div><h2>{t("Davis Catalogから追加")}</h2><p><code>{catalogTarget}</code> {t("へ追加するファイルを選択すると，自動で共有データ領域へダウンロードします．")}</p></div><button className="text-button" onClick={() => setCatalogTarget(undefined)}>{t("閉じる")}</button></div><input className="catalog-search" autoFocus value={catalogSearch} onChange={(event) => setCatalogSearch(event.target.value)} placeholder={t("データセット名，ファイル名，列名を検索")} /><div className="catalog-list">{catalogFiles.filter((file) => catalogMatch(file, catalogSearch)).slice(0, 100).map((file) => <button className="catalog-item" key={`${file.dataset_id}/${file.file_id}`} onClick={() => selectCatalogFile(file)}><strong>{file.title}</strong><span>{file.dataset_id} / {file.file_id}</span><small>{file.path} · {formatBytes(file.size)}{file.columns.length ? ` · ${file.columns.slice(0, 6).join(", ")}` : ""}</small></button>)}</div></div></div>}
    {busy && <div className="busy">{t("処理中です…")}</div>}
  </main>;
}

function SectionHeading({ number, title, description }: { number: string; title: string; description: string }) { return <div className="section-heading"><span>{number}</span><div><h2>{title}</h2><p>{description}</p></div></div>; }
function RunHistoryPanel({ entries, warnings, selected, comparison, busy, locale, onRefresh, onToggle, onOpen, onCompare }: { entries: RunHistoryEntry[]; warnings: string[]; selected: string[]; comparison: RunComparison[]; busy: boolean; locale: "ja" | "en"; onRefresh: () => void; onToggle: (runId: string) => void; onOpen: (entry: RunHistoryEntry) => void; onCompare: () => void }) {
  const { t } = useI18n();
  return <section><div className="heading-with-actions"><SectionHeading number="3" title="Run history" description={t("過去の実行を開き，標準artifact profileを使って比較します．")} /><div className="top-actions"><button className="secondary" disabled={busy} onClick={onRefresh}>{t("更新")}</button><button disabled={busy || selected.length < 2} onClick={onCompare}>{t("選択したRunを比較")} ({selected.length})</button></div></div>
    {!entries.length && <div className="empty-state">{t("このWorkspaceにはまだRunがありません．")}</div>}
    {warnings.length > 0 && <details className="history-warnings"><summary>{t("読み込めなかったRunがあります")} ({warnings.length})</summary><ul>{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></details>}
    <div className="run-history-list">{entries.map((entry) => { const runId = entry.run.request.run_id; return <article className={selected.includes(runId) ? "run-history-card selected" : "run-history-card"} key={runId}><label className="run-select"><input type="checkbox" checked={selected.includes(runId)} onChange={() => onToggle(runId)} /><span>{t("比較")}</span></label><div className="run-history-main"><strong>{entry.label || entry.plan_name}</strong><span>{runId}</span><small>{formatRunDate(entry.modified_at_unix_ms, locale)} · {entry.run.request.component.id} {entry.run.request.component.version} · {entry.run.request.operation}</small>{entry.tags.length > 0 && <div className="run-tags">{entry.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}</div><span className={`run-status ${entry.run.result.status}`}>{entry.run.result.status}</span><button className="secondary" onClick={() => onOpen(entry)}>{t("結果を見る")}</button></article>; })}</div>
    {comparison.length > 0 && <RunComparisonView runs={comparison} />}
  </section>;
}
function RunComparisonView({ runs }: { runs: RunComparison[] }) {
  const { t } = useI18n();
  const metricNames = [...new Set(runs.flatMap((run) => Object.keys(run.metrics)))].sort();
  const parameterNames = [...new Set(runs.flatMap((run) => Object.keys(run.parameters)))].sort();
  return <div className="run-comparison"><div className="subsection-heading"><div><h3>Run comparison</h3><p>{t("artifact名ではなくManifestに記録されたmetrics・parameters profileを照合しています．")}</p></div></div>
    {!metricNames.length && !parameterNames.length && <div className="empty-state">{t("選択したRunに比較可能なJSON／CSVの標準artifactがありません．")}</div>}
    {metricNames.length > 0 && <ComparisonTable title="Metrics" names={metricNames} runs={runs} value={(run, name) => run.metrics[name]} />}
    {parameterNames.length > 0 && <ComparisonTable title="Parameters (estimate)" names={parameterNames} runs={runs} value={(run, name) => run.parameters[name]?.estimate} />}
  </div>;
}
function ComparisonTable({ title, names, runs, value }: { title: string; names: string[]; runs: RunComparison[]; value: (run: RunComparison, name: string) => unknown }) {
  return <div className="comparison-table"><h4>{title}</h4><div className="result-table"><table><thead><tr><th>Name</th>{runs.map((run) => <th key={run.entry.run.request.run_id}>{run.entry.label || run.entry.plan_name}<small>{run.entry.run.request.component.id}</small></th>)}</tr></thead><tbody>{names.map((name) => <tr key={name}><th>{name}</th>{runs.map((run) => <td key={run.entry.run.request.run_id}>{formatResultValue(value(run, name))}</td>)}</tr>)}</tbody></table></div></div>;
}
function isComposedEditor(editor?: ComponentEditor) { return editor?.ui_schema.version === "davis.ui/v1"; }
function componentDisplayName(editor: ComponentEditor, locale: "ja" | "en") { return localizedText(editor.ui_schema.component_name, locale, editor.manifest.name); }
function defaultPlanName(editor: ComponentEditor) { return `${editor.manifest.id.split("/").pop() ?? "component"}-analysis`; }
function sourceId(path: string) { const raw = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? "data"; return raw.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^\d/, "data_$&") || "data"; }
function uniqueSourceId(path: string, existing: FormSource[]) { const base = sourceId(path); let candidate = base; let suffix = 2; while (existing.some((source) => source.id === candidate)) candidate = `${base}_${suffix++}`; return candidate; }
function serializeSource(source: FormSource, absolutePath: boolean): Record<string, unknown> { if (source.origin?.kind === "catalog") return { kind: "catalog", dataset_id: source.origin.dataset_id, file_id: source.origin.file_id, ...(source.origin.revision ? { revision: source.origin.revision } : {}) }; return { kind: "local", path: absolutePath ? source.path : source.serializedPath, ...(source.read ? { read: source.read } : {}) }; }
async function hydrateSource(key: string, source: PlanInput, resolved: Record<string, string>, t: (value: string) => string): Promise<FormSource> { const path = resolved[key]; if (!path) throw new Error(t("{id}の入力ファイルを解決できません．").replace("{id}", key)); if (source.kind === "local" && source.path) return { id: sourceId(source.path), path, serializedPath: source.path, read: source.read, origin: { kind: "local" }, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) }; if (source.kind === "catalog" && source.dataset_id && source.file_id) return { id: sourceId(source.file_id), path, serializedPath: source.file_id, origin: { kind: "catalog", dataset_id: source.dataset_id, file_id: source.file_id, revision: source.revision }, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) }; throw new Error(t("{id}はGUIで扱えない入力形式です．").replace("{id}", key)); }
function catalogMatch(file: CatalogFile, search: string) { const needle = search.trim().toLocaleLowerCase(); return !needle || [file.title, file.dataset_id, file.file_id, file.path, ...file.columns].some((value) => value.toLocaleLowerCase().includes(needle)); }
function formatBytes(size: number) { if (size < 1024) return `${size} B`; if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KiB`; return `${(size / 1024 ** 2).toFixed(1)} MiB`; }
function formatRunDate(unixMs: number, locale: "ja" | "en") { if (!unixMs) return locale === "ja" ? "日時不明" : "Unknown date"; return new Intl.DateTimeFormat(locale === "ja" ? "ja-JP" : "en-US", { dateStyle: "medium", timeStyle: "short" }).format(new Date(unixMs)); }
function PathField({ value, placeholder, onChange, onChoose }: { value: string; placeholder: string; onChange: (value: string) => void; onChoose: () => void }) { const { t } = useI18n(); return <div className="path-field"><input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} /><button className="secondary" onClick={onChoose}>{t("選択")}</button></div>; }
function resultDefinitions(editor: ComponentEditor | undefined, run: CompletedRun, locale: "ja" | "en" = "ja"): ResultDefinition[] {
  const explicit = (editor?.ui_schema.results ?? []).map((definition) => ({ ...definition, title: localizedText(definition.title, locale, definition.artifact), profile: editor?.manifest.outputs.artifacts[definition.artifact]?.profile })) as ResultDefinition[];
  const names = new Set(explicit.map((definition) => definition.artifact));
  const inferred = [...Object.entries(run.result.artifacts), ...Object.entries(run.result.extensions)].flatMap(([artifact, descriptor]): ResultDefinition[] => {
    if (names.has(artifact) || !descriptor.profile || !["application/json", "text/csv"].includes(descriptor.media_type)) return [];
    const table = ["table", "parameters", "predictions"].includes(descriptor.profile);
    return [{ artifact, title: profileTitle(descriptor.profile, locale), widget: table ? "table" : "key-value", profile: descriptor.profile }];
  });
  return [...explicit, ...inferred];
}
function profileTitle(profile: ArtifactProfile, locale: "ja" | "en") {
  const titles = locale === "ja" ? { table: "表", metrics: "指標", parameters: "推定parameter", predictions: "予測値", figure: "図", diagnostics: "診断", report: "Report" } : { table: "Table", metrics: "Metrics", parameters: "Parameters", predictions: "Predictions", figure: "Figure", diagnostics: "Diagnostics", report: "Report" };
  return (titles satisfies Record<ArtifactProfile, string>)[profile];
}
function ResultPreview({ definition, preview }: { definition: ResultDefinition; preview: ArtifactPreview }) {
  const { locale, t } = useI18n();
  if (definition.widget === "table" && isTablePreview(preview.content)) {
    const significanceIndex = preview.content.columns.indexOf("significance");
    const visibleColumns = preview.content.columns.map((column, index) => ({ column, index })).filter(({ column }) => definition.profile !== "parameters" || column !== "significance");
    return <article className="result-view"><div className="result-view-title"><h3>{definition.title}</h3><span>{definition.artifact}</span></div><div className="result-table"><table><thead><tr>{visibleColumns.map(({ column }) => <th key={column}>{parameterColumnLabel(column, definition.profile, locale)}</th>)}</tr></thead><tbody>{preview.content.rows.map((row, rowIndex) => <tr key={rowIndex}>{visibleColumns.map(({ column, index }) => <td key={index}>{formatParameterCell(column, row[index], definition.profile)}{definition.profile === "parameters" && column === "t_value" && significanceIndex >= 0 && row[significanceIndex] ? <sup>{row[significanceIndex]}</sup> : null}</td>)}</tr>)}</tbody></table>{preview.content.truncated && <p className="hint">{t("先頭200行を表示しています．")}</p>}</div></article>;
  }
  return <article className="result-view"><div className="result-view-title"><h3>{definition.title}</h3><span>{definition.artifact}</span></div><KeyValuePreview content={preview.content} profile={definition.profile} locale={locale} /></article>;
}
function KeyValuePreview({ content, profile, locale }: { content: unknown; profile?: ArtifactProfile; locale: "ja" | "en" }) { if (!content || typeof content !== "object" || Array.isArray(content)) return <pre>{formatResultValue(content)}</pre>; return <dl className="metric-grid">{Object.entries(content).map(([key, value]) => <div key={key}><dt>{profile === "metrics" ? metricLabel(key, locale) : key}</dt><dd>{profile === "metrics" ? formatMetricValue(key, value, locale) : formatResultValue(value)}</dd></div>)}</dl>; }
function isTablePreview(content: unknown): content is { columns: string[]; rows: string[][]; truncated: boolean } { if (!content || typeof content !== "object") return false; const candidate = content as { columns?: unknown; rows?: unknown }; return Array.isArray(candidate.columns) && Array.isArray(candidate.rows); }
function formatResultValue(value: unknown) { if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(6); if (typeof value === "string" || typeof value === "boolean") return String(value); if (value === null || value === undefined) return "—"; return JSON.stringify(value); }
function parameterColumnLabel(column: string, profile: ArtifactProfile | undefined, locale: "ja" | "en") { if (profile !== "parameters") return column; const labels = locale === "ja" ? { name: "説明変数", kind: "種類", estimate: "パラメータ", std_error: "標準誤差", t_value: "t値", p_value: "p値", fixed: "固定" } : { name: "Variable", kind: "Kind", estimate: "Parameter", std_error: "Std. error", t_value: "t value", p_value: "p value", fixed: "Fixed" }; return labels[column as keyof typeof labels] ?? column; }
function formatParameterCell(column: string, cell: string, profile?: ArtifactProfile) { if (profile !== "parameters" || !["estimate", "std_error", "t_value", "p_value"].includes(column)) return cell || "—"; const value = Number(cell); if (!cell || !Number.isFinite(value)) return "—"; return value.toFixed(column === "p_value" ? 3 : 2); }
function metricLabel(key: string, locale: "ja" | "en") { const labels = locale === "ja" ? { n_cases: "サンプル数 (ケース)", n_trips: "サンプル数 (trip)", n_rows: "データ行数", n_observed_links: "観測link数", n_parameters: "パラメータ数", null_model: "無情報モデル", inference_distribution: "推測統計の分布", inference_status: "推測統計の状態", log_likelihood_null: "初期尤度", log_likelihood_final: "最終尤度", rho_squared: "尤度比 ρ²", adjusted_rho_squared: "修正済み尤度比 ρ²", aic: "赤池情報量基準 (AIC)", bic: "ベイズ情報量基準 (BIC)", converged: "収束", iterations: "反復回数", message: "収束メッセージ" } : { n_cases: "Sample size (cases)", n_trips: "Sample size (trips)", n_rows: "Data rows", n_observed_links: "Observed links", n_parameters: "Parameters", null_model: "Null model", inference_distribution: "Inference distribution", inference_status: "Inference status", log_likelihood_null: "Initial log likelihood", log_likelihood_final: "Final log likelihood", rho_squared: "Likelihood ratio ρ²", adjusted_rho_squared: "Adjusted likelihood ratio ρ²", aic: "Akaike information criterion (AIC)", bic: "Bayesian information criterion (BIC)", converged: "Converged", iterations: "Iterations", message: "Convergence message" }; return labels[key as keyof typeof labels] ?? key; }
function formatMetricValue(key: string, value: unknown, locale: "ja" | "en") { if (value === null || value === undefined) return "—"; if (typeof value === "number") return ((Number.isInteger(value) && key.startsWith("n_")) || key === "iterations") ? String(value) : value.toFixed(2); if (typeof value === "boolean") return locale === "ja" ? value ? "はい" : "いいえ" : value ? "Yes" : "No"; if (value === "uniform_available_alternatives") return locale === "ja" ? "利用可能な選択肢を等確率" : "Uniform over available alternatives"; if (value === "uniform_feasible_outgoing_links") return locale === "ja" ? "目的地へ到達可能な出linkを等確率" : "Uniform over destination-feasible outgoing links"; if (value === "asymptotic_normal") return locale === "ja" ? "漸近正規分布" : "Asymptotic normal"; if (value === "available") return locale === "ja" ? "利用可能" : "Available"; if (value === "partial_boundary_or_singular_information") return locale === "ja" ? "一部利用不可 (境界値または特異な情報行列)" : "Partly unavailable (boundary or singular information)"; if (value === "unavailable_boundary_or_singular_information" || value === "unavailable_singular_information") return locale === "ja" ? "利用不可 (境界値または特異な情報行列)" : "Unavailable (boundary or singular information)"; if (value === "unavailable_not_converged") return locale === "ja" ? "利用不可 (未収束)" : "Unavailable (not converged)"; return formatResultValue(value); }
function comparisonMetrics(content: unknown, artifact: string): Record<string, unknown> {
  if (content && typeof content === "object" && !Array.isArray(content) && !isTablePreview(content)) return Object.fromEntries(Object.entries(content).map(([name, value]) => [artifact === "metrics" ? name : `${artifact}.${name}`, value]));
  if (!isTablePreview(content)) return {};
  const nameIndex = firstColumn(content.columns, ["name", "metric", "key"]); const valueIndex = firstColumn(content.columns, ["value", "estimate"]);
  if (nameIndex < 0 || valueIndex < 0) return {};
  return Object.fromEntries(content.rows.filter((row) => row[nameIndex]).map((row) => [row[nameIndex], row[valueIndex]]));
}
function comparisonParameters(content: unknown): Record<string, Record<string, string>> {
  if (!isTablePreview(content)) return {};
  const nameIndex = content.columns.indexOf("name"); if (nameIndex < 0) return {};
  return Object.fromEntries(content.rows.filter((row) => row[nameIndex]).map((row) => [row[nameIndex], Object.fromEntries(content.columns.map((column, index) => [column, row[index] ?? ""]))]));
}
function firstColumn(columns: string[], candidates: string[]) { return candidates.map((candidate) => columns.indexOf(candidate)).find((index) => index >= 0) ?? -1; }
