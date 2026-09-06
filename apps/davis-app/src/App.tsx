import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useEffect, useRef, useState } from "react";
import { bindingColumns, ColumnBinding, FormDefinition, FormInput, FormJoin, FormSource, JsonSchema, SchemaFormEditor } from "./SchemaFormEditor";

type ColumnProfile = { name: string; inferred_type: string; null_count: number; unique_sample: number; warnings: string[] };
type CsvProfile = { path: string; encoding: string; delimiter: string; rows_sampled: number; truncated: boolean; columns: ColumnProfile[] };
type Validation = { valid: boolean; plan: string; component: { id: string; version: string; manifest: string } };
type Artifact = { path: string; media_type: string; size?: number };
type CompletedRun = {
  run_directory: string; request: { run_id: string; component: { id: string; version: string } };
  result: { status: string; artifacts: Record<string, Artifact>; extensions: Record<string, Artifact> };
};
type ComponentEditor = {
  manifest: { id: string; name: string; version: string; kind: string; operations: string[]; inputs: Array<{ name: string; required: boolean }> };
  config_schema: JsonSchema; ui_schema: FormDefinition;
};
type PlanInput = {
  kind: string; path?: string; read?: unknown; processor?: { id: string; version: string }; sources?: Record<string, PlanInput>; base?: string;
  joins?: Array<{ source: string; left_on: string; right_on: string; relationship?: FormJoin["relationship"]; how?: FormJoin["how"]; allow_unmatched?: boolean }>;
  columns?: Record<string, ColumnBinding>;
};
type EditablePlan = {
  yaml: string; resolved_sources: Record<string, string>;
  plan: { name: string; component: { id: string; version: string; operation: string }; inputs: Record<string, PlanInput>; config: Record<string, unknown>; run?: { label?: string; tags?: string[]; [key: string]: unknown } };
  editor: ComponentEditor;
};
type ArtifactPreview = { name: string; media_type: string; content: unknown };

export default function App() {
  const [repository, setRepository] = useState(""); const [planPath, setPlanPath] = useState("");
  const [planName, setPlanName] = useState("analysis-plan"); const [runLabel, setRunLabel] = useState("");
  const [editor, setEditor] = useState<ComponentEditor>(); const [editorOptions, setEditorOptions] = useState<ComponentEditor[]>([]);
  const [inputs, setInputs] = useState<Record<string, FormInput | undefined>>({}); const [config, setConfig] = useState<Record<string, unknown>>({});
  const [preservedRun, setPreservedRun] = useState<Record<string, unknown>>({}); const [yamlPreview, setYamlPreview] = useState("");
  const [codeMode, setCodeMode] = useState(false); const [validation, setValidation] = useState<Validation>();
  const [completed, setCompleted] = useState<CompletedRun>(); const [artifactPreviews, setArtifactPreviews] = useState<Record<string, ArtifactPreview>>({});
  const [busy, setBusy] = useState(false); const [error, setError] = useState(""); const resultRef = useRef<HTMLElement>(null);

  useEffect(() => { if (completed) requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })); }, [completed]);
  async function perform(action: () => Promise<void>) { setBusy(true); setError(""); try { await action(); } catch (reason) { setError(String(reason)); } finally { setBusy(false); } }

  async function chooseRepository() {
    const selected = await open({ directory: true, multiple: false }); if (typeof selected !== "string") return;
    setRepository(selected);
    await perform(async () => {
      const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository: selected }); const definition = definitions[0];
      if (!definition) throw new Error("davis.ui/v1に対応するcomponentが見つかりません．");
      setEditorOptions(definitions); selectDefinition(definition);
    });
  }
  function selectDefinition(definition: ComponentEditor) {
    setEditor(definition); setPlanName(defaultPlanName(definition)); setRunLabel(""); setInputs({});
    setConfig(structuredClone(definition.ui_schema.defaults ?? {})); setPreservedRun({}); setPlanPath(""); setYamlPreview("");
    setCodeMode(false); setValidation(undefined); setCompleted(undefined); setArtifactPreviews({});
  }
  function selectEditor(identity: string) { const definition = editorOptions.find((item) => `${item.manifest.id}@${item.manifest.version}` === identity); if (definition) selectDefinition(definition); }
  async function newPlan() {
    await perform(async () => {
      let definition = editor;
      if (!definition || !isComposedEditor(definition)) { const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository }); setEditorOptions(definitions); definition = definitions[0]; }
      if (!definition) throw new Error("davis.ui/v1に対応するcomponentが見つかりません．"); selectDefinition(definition);
    });
  }

  async function addSources(slot: string) {
    const selected = await open({ multiple: true, filters: [{ name: "CSV", extensions: ["csv", "tsv"] }] });
    if (!Array.isArray(selected) || !selected.length) return;
    await perform(async () => {
      const inspected = await Promise.all(selected.map(async (path) => ({ path, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) })));
      setInputs((current) => {
        const existing = current[slot]; const additions: FormSource[] = [];
        for (const item of inspected) additions.push({ id: uniqueSourceId(item.path, [...(existing?.sources ?? []), ...additions]), path: item.path, serializedPath: item.path, profile: item.profile });
        const sources = [...(existing?.sources ?? []), ...additions];
        return { ...current, [slot]: { sources, base: existing?.base ?? sources[0].id, joins: existing?.joins ?? {}, columns: existing?.columns, processor: existing?.processor, forceBinding: existing?.forceBinding } };
      });
    });
  }

  function buildPlan(outputPath?: string): Record<string, unknown> {
    if (!editor || !isComposedEditor(editor)) throw new Error("このcomponentはdavis.ui/v1に対応していません．");
    const missing = editor.manifest.inputs.filter((item) => item.required).find((item) => !inputs[item.name]);
    if (missing) throw new Error(`${missing.name}の入力データを選択してください．`);
    const absolutePaths = Boolean(outputPath && planPath && outputPath !== planPath); const name = planName.trim() || "analysis-plan";
    return {
      api_version: "davis.analysis/v1alpha1", name,
      component: { id: editor.manifest.id, version: editor.manifest.version, operation: editor.manifest.operations.includes("estimate") ? "estimate" : editor.manifest.operations[0] },
      inputs: Object.fromEntries(Object.entries(inputs).filter(([, input]) => input).map(([slot, input]) => [slot, buildInputBinding(slot, input!, absolutePaths)])),
      config, run: buildRunMetadata(["gui", editor.manifest.id.replace("/", "-")]),
    };
  }
  function buildInputBinding(slot: string, input: FormInput, absolutePaths: boolean): Record<string, unknown> {
    if (input.sources.length === 1 && !input.forceBinding) { const source = input.sources[0]; return { kind: "local", path: absolutePaths ? source.path : source.serializedPath, ...(source.read ? { read: source.read } : {}) }; }
    const joins = input.sources.filter((source) => source.id !== input.base).map((source) => { const join = input.joins[source.id]; if (!join?.leftOn || !join.rightOn) throw new Error(`${slot}の${source.id}について結合キーを選択してください．`); return { source: source.id, left_on: join.leftOn, right_on: join.rightOn, relationship: join.relationship, how: join.how, allow_unmatched: join.allowUnmatched }; });
    const preparation = editor?.ui_schema.inputs?.[slot]?.preparation;
    return { kind: "table_binding", processor: input.processor ?? (preparation ? { id: preparation.component, version: preparation.version } : { id: "davis/csv-transform", version: "0.4.0" }),
      sources: Object.fromEntries(input.sources.map((source) => [source.id, { kind: "local", path: absolutePaths ? source.path : source.serializedPath, ...(source.read ? { read: source.read } : {}) }])),
      base: input.base, joins, columns: Object.fromEntries(bindingColumns(input).map(({ alias, source, column }) => [alias, { source, column }])) };
  }
  function buildRunMetadata(defaultTags: string[]) { const metadata = { ...preservedRun }; delete metadata.label; if (runLabel.trim()) metadata.label = runLabel.trim(); if (!Array.isArray(metadata.tags)) metadata.tags = defaultTags; return metadata; }

  async function previewPlan() { await perform(async () => setYamlPreview(await invoke<string>("render_analysis_plan", { plan: buildPlan() }))); }
  async function saveDraft(execute: boolean, overwrite = false) {
    await perform(async () => {
      const target = overwrite ? planPath : await save({ defaultPath: repository ? `${repository}/model.yaml` : "model.yaml", filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] });
      if (!target) return; const plan = buildPlan(target); const saved = await invoke<string>("save_analysis_plan", { repository, path: target, plan });
      setPlanPath(saved); setYamlPreview(await invoke<string>("render_analysis_plan", { plan })); setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: saved })); setCompleted(undefined);
      if (execute) await showCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: saved }));
    });
  }
  async function openPlanForEditing() {
    if (!repository) return setError("先にWorkspaceを選択してください．");
    const selected = await open({ multiple: false, filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] }); if (typeof selected !== "string") return;
    await perform(async () => {
      const loaded = await invoke<EditablePlan>("load_analysis_plan_for_editing", { repository, path: selected });
      setEditorOptions((current) => current.some((item) => item.manifest.id === loaded.editor.manifest.id && item.manifest.version === loaded.editor.manifest.version) ? current : [...current, loaded.editor]);
      if (isComposedEditor(loaded.editor)) await hydratePlan(loaded, selected); else { setEditor(loaded.editor); setPlanPath(selected); setRunLabel(loaded.plan.run?.label ?? ""); setYamlPreview(loaded.yaml); setCodeMode(true); setCompleted(undefined); setArtifactPreviews({}); setValidation(undefined); }
    });
  }
  async function hydratePlan(loaded: EditablePlan, path: string) {
    const loadedInputs: Record<string, FormInput | undefined> = {}; const issues: string[] = [];
    setEditor(loaded.editor); setPlanName(loaded.plan.name); setRunLabel(loaded.plan.run?.label ?? ""); setConfig(structuredClone(loaded.plan.config)); setPreservedRun(loaded.plan.run ?? {});
    setPlanPath(path); setYamlPreview(loaded.yaml); setCodeMode(false); setValidation(undefined); setCompleted(undefined); setArtifactPreviews({});
    for (const declaration of loaded.editor.manifest.inputs) { const input = loaded.plan.inputs[declaration.name]; if (!input) continue; try { loadedInputs[declaration.name] = await hydrateInputBinding(declaration.name, input, loaded.resolved_sources); } catch (reason) { issues.push(`${declaration.name}を読み込めませんでした: ${String(reason)}`); } }
    setInputs(loadedInputs); if (issues.length) setError(issues.join("\n"));
  }
  async function hydrateInputBinding(slot: string, input: PlanInput, resolved: Record<string, string>): Promise<FormInput> {
    if (input.kind === "local" && input.path) { const path = resolved[slot]; if (!path) throw new Error("local pathを解決できません．"); const id = sourceId(input.path); return { sources: [{ id, path, serializedPath: input.path, read: input.read, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) }], base: id, joins: {} }; }
    if (input.kind !== "table_binding" || !input.sources || !input.base || !input.columns) throw new Error("GUIで扱えない入力形式です．");
    const sources: FormSource[] = await Promise.all(Object.entries(input.sources).map(async ([id, source]) => { if (source.kind !== "local" || !source.path) throw new Error(`${id}はlocal CSVではありません．`); const path = resolved[`${slot}/${id}`] ?? resolved[id]; if (!path) throw new Error(`${id}のlocal pathを解決できません．`); return { id, path, serializedPath: source.path, read: source.read, profile: await invoke<CsvProfile>("inspect_csv_file", { path }) }; }));
    const joins = Object.fromEntries((input.joins ?? []).map((join) => [join.source, { leftOn: join.left_on, rightOn: join.right_on, relationship: join.relationship ?? "many_to_one", how: join.how ?? "left", allowUnmatched: join.allow_unmatched ?? false } satisfies FormJoin]));
    return { sources, base: input.base, joins, columns: input.columns, processor: input.processor, forceBinding: true };
  }

  async function saveCodePlan(execute: boolean) { await perform(async () => { if (!planPath) throw new Error("保存先がありません．既存Planを開き直してください．"); await invoke<string>("save_analysis_plan_yaml", { repository, path: planPath, yaml: yamlPreview }); setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: planPath })); if (execute) await showCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: planPath })); }); }
  async function showCompleted(run: CompletedRun) {
    setCompleted(run); const definitions = editor?.ui_schema.results ?? [];
    const previews = await Promise.all(definitions.map(async (definition) => { try { return await invoke<ArtifactPreview>("preview_run_artifact", { repository, runId: run.request.run_id, artifact: definition.artifact }); } catch { return undefined; } }));
    setArtifactPreviews(Object.fromEntries(previews.filter(Boolean).map((preview) => [preview!.name, preview!])));
  }
  async function openRunDirectory() { if (completed) await perform(async () => invoke("open_run_directory", { repository, runId: completed.request.run_id })); }

  const editorReady = repository.length > 0 && (editor?.manifest.inputs.filter((item) => item.required).every((item) => Boolean(inputs[item.name]?.sources.length)) ?? false);
  return <main><header><p className="eyebrow">DAVIS MODEL</p><h1>ローカルデータから推定まで</h1><p className="lead">ComponentManifestに従い，入力結合とモデル設定を同じAnalysisPlanとして編集します．</p></header>
    {error && <div className="error sticky-error">{error}</div>}
    <section><SectionHeading number="1" title="Project workspace" description="model.yamlとdavis-runsを置く作業folderです．Davis repositoryのcloneは不要です．" /><PathField value={repository} placeholder="Project folder" onChange={(value) => { setRepository(value); setEditor(undefined); }} onChoose={chooseRepository} /></section>
    <section><div className="heading-with-actions"><SectionHeading number="2" title="Analysis plan editor" description="すべてのcomponentを同じdavis.ui/v1 rendererで編集します．" /><div className="top-actions"><button className="secondary" onClick={newPlan}>新規Plan</button><button className="secondary" disabled={!repository} onClick={openPlanForEditing}>既存Planを開く</button></div></div>
      <div className="field-grid compact-grid"><label><span>Plan name</span><input value={planName} onChange={(event) => setPlanName(event.target.value)} /></label><label><span>Run name (folder prefix)</span><input value={runLabel} disabled={codeMode} placeholder="空欄ならPlan name" onChange={(event) => setRunLabel(event.target.value)} /></label><label><span>ComponentManifest</span><select value={editor ? `${editor.manifest.id}@${editor.manifest.version}` : ""} disabled={!editorOptions.length || codeMode} onChange={(event) => selectEditor(event.target.value)}>{!editor && <option value="">Workspaceを選択してください</option>}{editorOptions.map((item) => <option key={`${item.manifest.id}@${item.manifest.version}`} value={`${item.manifest.id}@${item.manifest.version}`}>{item.manifest.name} ({item.manifest.id} {item.manifest.version})</option>)}</select></label></div>
      {codeMode && <div className="code-mode"><div className="notice">このcomponentにはdavis.ui/v1の画面定義がありません．内容を失わないYAML modeで開いています．</div><textarea className="yaml-preview editable" value={yamlPreview} onChange={(event) => setYamlPreview(event.target.value)} aria-label="model.yaml code editor" /><div className="actions"><button className="secondary" disabled={busy} onClick={() => saveCodePlan(false)}>上書き保存・検証</button><button disabled={busy} onClick={() => saveCodePlan(true)}>上書きして実行</button></div></div>}
      {!codeMode && editor && isComposedEditor(editor) && <><SchemaFormEditor definition={editor} inputs={inputs} config={config} onAddSources={addSources} onInputChange={(slot, input) => setInputs((current) => ({ ...current, [slot]: input }))} onConfigChange={setConfig} />
        <div className="actions editor-actions"><button className="secondary" disabled={!editorReady || busy} onClick={previewPlan}>YAMLを確認</button><button className="secondary" disabled={!editorReady || busy} onClick={() => saveDraft(false)}>別名で保存</button><button className="secondary" disabled={!editorReady || !planPath || busy} onClick={() => saveDraft(false, true)}>上書き保存</button><button disabled={!editorReady || busy} onClick={() => saveDraft(true, Boolean(planPath))}>{planPath ? "上書きして推定" : "保存して推定"}</button></div>
        {validation && <div className="success">{validation.component.id} {validation.component.version}として保存・検証しました．</div>}{planPath && <div className="plan-path">{planPath}</div>}{yamlPreview && <textarea className="yaml-preview" readOnly value={yamlPreview} aria-label="生成されたmodel.yaml" />}</>}
    </section>
    {completed && <section ref={resultRef}><SectionHeading number="3" title="Run result" description={completed.request.run_id} /><div className="result-views">{(editor?.ui_schema.results ?? []).map((definition) => { const preview = artifactPreviews[definition.artifact]; return preview ? <ResultPreview key={definition.artifact} definition={definition} preview={preview} /> : null; })}</div><div className="run-directory-row"><div className="run-directory">{completed.run_directory}</div><button className="secondary" disabled={busy} onClick={openRunDirectory}>結果フォルダを開く</button></div><div className="artifacts">{[...Object.entries(completed.result.artifacts), ...Object.entries(completed.result.extensions)].map(([name, artifact]) => <article key={name}><strong>{name}</strong><span>{artifact.path}</span><small>{artifact.media_type}{artifact.size ? ` · ${artifact.size} bytes` : ""}</small></article>)}</div></section>}
    {busy && <div className="busy">処理中です…</div>}
  </main>;
}

function SectionHeading({ number, title, description }: { number: string; title: string; description: string }) { return <div className="section-heading"><span>{number}</span><div><h2>{title}</h2><p>{description}</p></div></div>; }
function isComposedEditor(editor?: ComponentEditor) { return editor?.ui_schema.version === "davis.ui/v1"; }
function defaultPlanName(editor: ComponentEditor) { return `${editor.manifest.id.split("/").pop() ?? "component"}-analysis`; }
function sourceId(path: string) { const raw = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? "data"; return raw.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^\d/, "data_$&") || "data"; }
function uniqueSourceId(path: string, existing: FormSource[]) { const base = sourceId(path); let candidate = base; let suffix = 2; while (existing.some((source) => source.id === candidate)) candidate = `${base}_${suffix++}`; return candidate; }
function PathField({ value, placeholder, onChange, onChoose }: { value: string; placeholder: string; onChange: (value: string) => void; onChoose: () => void }) { return <div className="path-field"><input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} /><button className="secondary" onClick={onChoose}>選択</button></div>; }
function ResultPreview({ definition, preview }: { definition: { artifact: string; title: string; widget: "key-value" | "table" }; preview: ArtifactPreview }) { return <article className="result-view"><div className="result-view-title"><h3>{definition.title}</h3><span>{definition.artifact}</span></div>{definition.widget === "table" && isTablePreview(preview.content) ? <div className="result-table"><table><thead><tr>{preview.content.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview.content.rows.map((row, index) => <tr key={index}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table>{preview.content.truncated && <p className="hint">先頭200行を表示しています．</p>}</div> : <KeyValuePreview content={preview.content} />}</article>; }
function KeyValuePreview({ content }: { content: unknown }) { if (!content || typeof content !== "object" || Array.isArray(content)) return <pre>{formatResultValue(content)}</pre>; return <dl className="metric-grid">{Object.entries(content).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{formatResultValue(value)}</dd></div>)}</dl>; }
function isTablePreview(content: unknown): content is { columns: string[]; rows: string[][]; truncated: boolean } { if (!content || typeof content !== "object") return false; const candidate = content as { columns?: unknown; rows?: unknown }; return Array.isArray(candidate.columns) && Array.isArray(candidate.rows); }
function formatResultValue(value: unknown) { if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(6); if (typeof value === "string" || typeof value === "boolean") return String(value); if (value === null || value === undefined) return "—"; return JSON.stringify(value); }
