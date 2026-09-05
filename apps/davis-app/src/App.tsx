import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useState } from "react";

type ColumnProfile = { name: string; inferred_type: string; null_count: number; unique_sample: number; warnings: string[] };
type CsvProfile = { path: string; encoding: string; delimiter: string; rows_sampled: number; truncated: boolean; columns: ColumnProfile[] };
type Validation = { valid: boolean; plan: string; component: { id: string; version: string; manifest: string } };
type Artifact = { path: string; media_type: string; size?: number };
type CompletedRun = {
  run_directory: string;
  request: { run_id: string; component: { id: string; version: string } };
  result: { status: string; artifacts: Record<string, Artifact>; extensions: Record<string, Artifact> };
};
type SourceDraft = { id: string; path: string; serializedPath: string; read?: unknown; profile: CsvProfile };
type ColumnRef = { source: string; column: string };
type RoleName = string;
type JoinDraft = {
  leftOn: string; rightOn: string; relationship: "many_to_one" | "one_to_one";
  how: "left" | "inner"; allowUnmatched: boolean;
};
type TermDraft = {
  id: number; parameter: string; mode: "column" | "constant"; source: string;
  column: string; constant: string; alternatives: string;
};
type ComponentEditor = {
  manifest: { id: string; name: string; version: string; kind: string; operations: string[] };
  config_schema: {
    properties?: { roles?: { required?: string[]; properties?: Record<string, unknown> } };
  };
  ui_schema: {
    "ui:editor"?: string;
    "ui:inputPreparation"?: { component: string; version: string };
    roles?: { "ui:labels"?: Record<string, string> };
    terms?: { "ui:alternativesFromRole"?: string };
  };
};
type PlanInput = {
  kind: string;
  path?: string;
  read?: unknown;
  processor?: { id: string; version: string };
  sources?: Record<string, PlanInput>;
  base?: string;
  joins?: Array<{ source: string; left_on: string; right_on: string; relationship?: JoinDraft["relationship"]; how?: JoinDraft["how"]; allow_unmatched?: boolean }>;
  columns?: Record<string, ColumnRef>;
};
type EditablePlan = {
  yaml: string;
  resolved_sources: Record<string, string>;
  plan: {
    name: string;
    component: { id: string; version: string; operation: string };
    inputs: Record<string, PlanInput>;
    config: { roles?: Record<string, string>; terms?: Array<{ parameter: string; column?: string; constant?: number; alternatives?: Array<string | number> }>; [key: string]: unknown };
    run?: { label?: string; tags?: string[] };
  };
  editor: ComponentEditor;
};
type DistinctValues = { values: string[]; rows_sampled: number; truncated: boolean };

const defaultRoleLabels: Record<string, string> = {
  case_id: "ケースID", alternative_id: "選択肢ID", chosen: "選択結果",
  available: "利用可能性 (任意)", weight: "重み (任意)",
};

export default function App() {
  const [repository, setRepository] = useState("");
  const [planPath, setPlanPath] = useState("");
  const [editor, setEditor] = useState<ComponentEditor>();
  const [editorOptions, setEditorOptions] = useState<ComponentEditor[]>([]);
  const [validation, setValidation] = useState<Validation>();
  const [completed, setCompleted] = useState<CompletedRun>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [planName, setPlanName] = useState("mode-choice-model");
  const [sources, setSources] = useState<SourceDraft[]>([]);
  const [baseSource, setBaseSource] = useState("");
  const [joins, setJoins] = useState<Record<string, JoinDraft>>({});
  const [roles, setRoles] = useState<Record<RoleName, ColumnRef | undefined>>({});
  const [terms, setTerms] = useState<TermDraft[]>([]);
  const [nextTermId, setNextTermId] = useState(1);
  const [yamlPreview, setYamlPreview] = useState("");
  const [codeMode, setCodeMode] = useState(false);
  const [alternativeValues, setAlternativeValues] = useState<DistinctValues>();
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});
  const [preservedConfig, setPreservedConfig] = useState<Record<string, unknown>>({});
  const [preservedRun, setPreservedRun] = useState<Record<string, unknown>>({});
  const [preparation, setPreparation] = useState<{ component: string; version: string }>();

  async function perform(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); } catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  }

  async function chooseRepository() {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected !== "string") return;
    setRepository(selected);
    await perform(async () => {
      const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository: selected });
      const definition = definitions[0];
      if (!definition) throw new Error("Form editorに対応するcomponentが見つかりません．");
      setEditorOptions(definitions);
      setEditor(definition);
      setPreparation(definition.ui_schema["ui:inputPreparation"]);
      initializeRoles(definition);
    });
  }

  function requireSupportedEditor(definition: ComponentEditor) {
    if (definition.ui_schema["ui:editor"] !== "linear-utility") {
      throw new Error(`${definition.manifest.id}はこのGUIのForm editorに対応していません．YAML modeを利用してください．`);
    }
  }

  function initializeRoles(definition: ComponentEditor) {
    const names = Object.keys(definition.config_schema.properties?.roles?.properties ?? {});
    setRoles(Object.fromEntries(names.map((name) => [name, undefined])));
  }

  async function addDataSources() {
    const selected = await open({ multiple: true, filters: [{ name: "CSV", extensions: ["csv", "tsv"] }] });
    const paths = typeof selected === "string" ? [selected] : selected;
    if (!paths?.length) return;
    await perform(async () => {
      const profiles = await Promise.all(paths.filter((path) => !sources.some((source) => source.path === path))
        .map((path) => invoke<CsvProfile>("inspect_csv_file", { path })));
      const used = new Set(sources.map((source) => source.id));
      const added = profiles.map((newProfile) => {
        const raw = newProfile.path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? "data";
        const stem = raw.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^\d/, "data_$&") || "data";
        let id = stem; let suffix = 2;
        while (used.has(id)) id = `${stem}_${suffix++}`;
        used.add(id);
        return { id, path: newProfile.path, serializedPath: newProfile.path, profile: newProfile };
      });
      const updated = [...sources, ...added];
      setSources(updated);
      if (!baseSource && updated[0]) {
        setBaseSource(updated[0].id); suggestRoles(updated[0]);
        const alternative = ["alternative", "alternative_id", "alt_id", "alt"]
          .find((name) => updated[0].profile.columns.some((column) => column.name === name));
        if (alternative) setAlternativeValues(await invoke<DistinctValues>("inspect_distinct_values", {
          path: updated[0].path, column: alternative,
        }));
      }
    });
  }

  function suggestRoles(source: SourceDraft) {
    const names = new Set(source.profile.columns.map((column) => column.name));
    const candidates: Record<RoleName, string[]> = {
      case_id: ["case_id", "choice_id", "person_id", "id"],
      alternative_id: ["alternative", "alternative_id", "alt_id", "alt"],
      chosen: ["chosen", "choice", "selected"], available: ["available", "availability"],
    };
    setRoles((current) => {
      const updated = { ...current };
      (Object.keys(candidates) as RoleName[]).forEach((role) => {
        const column = candidates[role].find((name) => names.has(name));
        if (!updated[role] && column) updated[role] = { source: source.id, column };
      });
      return updated;
    });
  }

  function removeSource(id: string) {
    const updated = sources.filter((source) => source.id !== id);
    setSources(updated);
    setJoins((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== id)));
    setRoles((current) => Object.fromEntries(Object.entries(current)
      .map(([role, ref]) => [role, ref?.source === id ? undefined : ref])) as Record<RoleName, ColumnRef | undefined>);
    setTerms((current) => current.map((term) => term.source === id ? { ...term, source: "", column: "" } : term));
    if (baseSource === id) { setBaseSource(updated[0]?.id ?? ""); if (updated[0]) suggestRoles(updated[0]); }
  }

  function joinFor(sourceId: string): JoinDraft {
    return joins[sourceId] ?? { leftOn: "", rightOn: "", relationship: "many_to_one", how: "left", allowUnmatched: false };
  }
  function updateJoin(sourceId: string, patch: Partial<JoinDraft>) {
    setJoins((current) => ({ ...current, [sourceId]: { ...joinFor(sourceId), ...patch } }));
  }
  function addTerm() {
    setTerms((current) => [...current, {
      id: nextTermId, parameter: `beta_${nextTermId}`, mode: "column", source: baseSource,
      column: "", constant: "1", alternatives: "",
    }]);
    setNextTermId((current) => current + 1);
  }
  function updateTerm(id: number, patch: Partial<TermDraft>) {
    setTerms((current) => current.map((term) => term.id === id ? { ...term, ...patch } : term));
  }

  async function selectRole(role: RoleName, ref: ColumnRef | undefined) {
    setRoles((current) => ({ ...current, [role]: ref }));
    const alternativeRole = editor?.ui_schema.terms?.["ui:alternativesFromRole"];
    if (role !== alternativeRole || !ref) {
      if (role === alternativeRole) setAlternativeValues(undefined);
      return;
    }
    const source = sources.find((item) => item.id === ref.source);
    if (!source) return;
    await perform(async () => {
      setAlternativeValues(await invoke<DistinctValues>("inspect_distinct_values", {
        path: source.path, column: ref.column,
      }));
    });
  }

  function buildPlan(outputPath?: string): Record<string, unknown> {
    if (!editor) throw new Error("Workspaceを選択してComponentManifestを読み込んでください．");
    if (!sources.length) throw new Error("少なくとも1つのCSVを追加してください．");
    if (!baseSource) throw new Error("基準データを選択してください．");
    const requiredRoles = editor.config_schema.properties?.roles?.required ?? [];
    const missingRole = requiredRoles.find((role) => !roles[role]);
    if (missingRole) throw new Error(`${roleLabel(missingRole, editor)}の列を選択してください．`);
    if (!terms.length) throw new Error("説明変数または定数項を1つ以上追加してください．");
    for (const term of terms) {
      if (!term.parameter.trim()) throw new Error("すべてのtermにparameter名を入力してください．");
      if (term.mode === "column" && (!term.source || !term.column)) throw new Error(`${term.parameter}の参照列を選択してください．`);
      if (term.mode === "constant" && !Number.isFinite(Number(term.constant))) throw new Error(`${term.parameter}の定数を数値で入力してください．`);
    }

    const refs: ColumnRef[] = Object.values(roles).filter(Boolean) as ColumnRef[];
    terms.filter((term) => term.mode === "column").forEach((term) => refs.push({ source: term.source, column: term.column }));
    const uniqueRefs = [...new Map(refs.map((ref) => [`${ref.source}\0${ref.column}`, ref])).values()];
    const counts = uniqueRefs.reduce<Record<string, number>>((result, ref) => ({ ...result, [ref.column]: (result[ref.column] ?? 0) + 1 }), {});
    const aliases = new Map(uniqueRefs.map((ref) => [`${ref.source}\0${ref.column}`, counts[ref.column] > 1 ? `${ref.source}_${ref.column}` : ref.column]));
    const aliasFor = (ref: ColumnRef) => aliases.get(`${ref.source}\0${ref.column}`)!;
    const saveAsDifferentFile = Boolean(outputPath && planPath && outputPath !== planPath);
    const sourceMap: Record<string, Record<string, unknown>> = Object.fromEntries(sources.map((source) => [source.id, {
      kind: "local", path: saveAsDifferentFile ? source.path : source.serializedPath,
      ...(source.read ? { read: source.read } : {}),
    }]));
    let choiceData: Record<string, unknown> = sourceMap[sources[0].id];
    if (sources.length > 1) {
      const joinList = sources.filter((source) => source.id !== baseSource).map((source) => {
        const join = joinFor(source.id);
        if (!join.leftOn || !join.rightOn) throw new Error(`${source.id}の結合キーを選択してください．`);
        return { source: source.id, left_on: join.leftOn, right_on: join.rightOn, relationship: join.relationship, how: join.how, allow_unmatched: join.allowUnmatched };
      });
      choiceData = {
        kind: "table_binding", processor: {
          id: preparation?.component ?? editor.ui_schema["ui:inputPreparation"]?.component ?? "davis/csv-transform",
          version: preparation?.version ?? editor.ui_schema["ui:inputPreparation"]?.version ?? "0.4.0",
        },
        sources: sourceMap, base: baseSource, joins: joinList,
        columns: Object.fromEntries(uniqueRefs.map((ref) => [aliasFor(ref), ref])),
      };
    }
    const configRoles = Object.fromEntries((Object.entries(roles) as [RoleName, ColumnRef | undefined][])
      .filter(([, ref]) => ref).map(([role, ref]) => [role, aliasFor(ref!)]));
    const configTerms = terms.map((term) => {
      const alternatives = term.alternatives.split(",").map((value) => value.trim()).filter(Boolean);
      return {
        parameter: term.parameter.trim(),
        ...(term.mode === "column" ? { column: aliasFor({ source: term.source, column: term.column }) } : { constant: Number(term.constant) }),
        ...(alternatives.length ? { alternatives } : {}),
      };
    });
    const name = planName.trim() || "mode-choice-model";
    return {
      api_version: "davis.analysis/v1alpha1", name,
      component: { id: editor.manifest.id, version: editor.manifest.version, operation: "estimate" },
      inputs: { choice_data: choiceData },
      config: {
        ...preservedConfig, roles: configRoles, terms: configTerms,
        estimation: preservedConfig.estimation ?? { optimizer: "bfgs", max_iterations: 500, tolerance: 1e-8 },
      },
      run: Object.keys(preservedRun).length ? preservedRun : { label: name, tags: ["gui", "mnl"] },
    };
  }

  async function previewPlan() {
    await perform(async () => setYamlPreview(await invoke<string>("render_analysis_plan", { plan: buildPlan() })));
  }
  async function saveDraft(execute: boolean, overwrite = false) {
    await perform(async () => {
      const target = overwrite ? planPath : await save({
        defaultPath: repository ? `${repository}/davis-runs/model.yaml` : "model.yaml",
        filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }],
      });
      if (!target) return;
      const plan = buildPlan(target);
      const saved = await invoke<string>("save_analysis_plan", { repository, path: target, plan });
      setPlanPath(saved); setYamlPreview(await invoke<string>("render_analysis_plan", { plan }));
      setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: saved }));
      setCompleted(undefined);
      if (execute) setCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: saved }));
    });
  }
  async function openPlanForEditing() {
    if (!repository) throw new Error("先にWorkspaceを選択してください．");
    const selected = await open({ multiple: false, filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] });
    if (typeof selected !== "string") return;
    await perform(async () => {
      const loaded = await invoke<EditablePlan>("load_analysis_plan_for_editing", { repository, path: selected });
      setEditorOptions((current) => current.some((item) => item.manifest.id === loaded.editor.manifest.id && item.manifest.version === loaded.editor.manifest.version)
        ? current : [...current, loaded.editor]);
      if (loaded.editor.ui_schema["ui:editor"] !== "linear-utility") {
        setEditor(loaded.editor); setPlanPath(selected); setYamlPreview(loaded.yaml); setCodeMode(true);
        setSources([]); setCompleted(undefined); setValidation(undefined);
        return;
      }
      await hydratePlan(loaded, selected);
    });
  }

  async function hydratePlan(loaded: EditablePlan, path: string) {
    requireSupportedEditor(loaded.editor);
    const input = loaded.plan.inputs.choice_data ?? Object.values(loaded.plan.inputs)[0];
    if (!input) throw new Error("Planに入力データがありません．");
    let sourceInputs: Record<string, PlanInput>;
    let loadedBase: string;
    let aliasToRef: Record<string, ColumnRef>;
    let loadedJoins: Record<string, JoinDraft> = {};
    if (input.kind === "local" && input.path) {
      loadedBase = sourceId(input.path);
      sourceInputs = { [loadedBase]: input };
      aliasToRef = {};
    } else if (input.kind === "table_binding" && input.sources && input.base && input.columns) {
      sourceInputs = input.sources; loadedBase = input.base; aliasToRef = input.columns;
      for (const join of input.joins ?? []) {
        if (typeof join.left_on !== "string" || typeof join.right_on !== "string") {
          throw new Error("複合キーを持つPlanはまだFormへ読み戻せません．YAML modeでは実行できます．");
        }
        loadedJoins[join.source] = {
          leftOn: join.left_on, rightOn: join.right_on,
          relationship: join.relationship ?? "many_to_one", how: join.how ?? "left",
          allowUnmatched: join.allow_unmatched ?? false,
        };
      }
    } else {
      throw new Error("この入力形式はまだFormへ読み戻せません．YAML modeでは実行できます．");
    }
    const loadedSources = await Promise.all(Object.entries(sourceInputs).map(async ([id, source]) => {
      if (source.kind !== "local" || !source.path) throw new Error(`${id}はlocal CSVではないためFormへ読み戻せません．`);
      const resolved = loaded.resolved_sources[id]
        ?? (Object.keys(sourceInputs).length === 1 ? Object.values(loaded.resolved_sources)[0] : undefined);
      if (!resolved) throw new Error(`${id}のlocal pathを解決できません．`);
      return {
        id, path: resolved, serializedPath: source.path, read: source.read,
        profile: await invoke<CsvProfile>("inspect_csv_file", { path: resolved }),
      };
    }));
    if (input.kind === "local") {
      const only = loadedSources[0];
      aliasToRef = Object.fromEntries(only.profile.columns.map((column) => [column.name, { source: only.id, column: column.name }]));
    }
    const loadedRoles = Object.fromEntries(Object.entries(loaded.plan.config.roles ?? {})
      .map(([role, alias]) => [role, aliasToRef[alias]]));
    const loadedTerms = (loaded.plan.config.terms ?? []).map((term, index) => {
      const ref = term.column ? aliasToRef[term.column] : undefined;
      if (term.column && !ref) throw new Error(`term ${term.parameter}の列 ${term.column}を入力bindingへ対応付けられません．`);
      return {
        id: index + 1, parameter: term.parameter, mode: term.column ? "column" as const : "constant" as const,
        source: ref?.source ?? "", column: ref?.column ?? "", constant: String(term.constant ?? 1),
        alternatives: (term.alternatives ?? []).map(String).join(", "),
      };
    });
    setEditor(loaded.editor); setPlanName(loaded.plan.name); setSources(loadedSources);
    setPreparation(input.processor
      ? { component: input.processor.id, version: input.processor.version }
      : loaded.editor.ui_schema["ui:inputPreparation"]);
    const { roles: ignoredRoles, terms: ignoredTerms, ...remainingConfig } = loaded.plan.config;
    void ignoredRoles; void ignoredTerms;
    setPreservedConfig(remainingConfig); setPreservedRun(loaded.plan.run ?? {});
    setBaseSource(loadedBase); setJoins(loadedJoins); setRoles(loadedRoles); setTerms(loadedTerms);
    setNextTermId(loadedTerms.length + 1); setPlanPath(path); setValidation(undefined); setCompleted(undefined);
    setYamlPreview(loaded.yaml); setCodeMode(false);
    const alternativeRole = loaded.editor.ui_schema.terms?.["ui:alternativesFromRole"];
    const alternativeRef = alternativeRole ? loadedRoles[alternativeRole] : undefined;
    const alternativeSource = loadedSources.find((source) => source.id === alternativeRef?.source);
    setAlternativeValues(alternativeRef && alternativeSource
      ? await invoke<DistinctValues>("inspect_distinct_values", { path: alternativeSource.path, column: alternativeRef.column })
      : undefined);
  }

  async function newPlan() {
    await perform(async () => {
      let definition = editor;
      if (!definition || definition.ui_schema["ui:editor"] !== "linear-utility") {
        const definitions = await invoke<ComponentEditor[]>("component_editor_definitions", { repository });
        definition = definitions[0]; setEditorOptions(definitions);
        if (!definition) throw new Error("Form editorに対応するcomponentが見つかりません．");
        setEditor(definition);
      }
      setPlanPath(""); setPlanName("mode-choice-model"); setSources([]); setBaseSource(""); setJoins({});
      if (definition) initializeRoles(definition); else setRoles({});
      setTerms([]); setNextTermId(1); setYamlPreview(""); setCodeMode(false); setAlternativeValues(undefined);
      setPreservedConfig({}); setPreservedRun({});
      setPreparation(definition?.ui_schema["ui:inputPreparation"]);
      setValidation(undefined); setCompleted(undefined);
    });
  }
  function selectEditor(identity: string) {
    const definition = editorOptions.find((item) => `${item.manifest.id}@${item.manifest.version}` === identity);
    if (!definition) return;
    setEditor(definition); initializeRoles(definition); setPreparation(definition.ui_schema["ui:inputPreparation"]);
    setTerms([]); setYamlPreview(""); setPlanPath(""); setValidation(undefined); setCompleted(undefined);
  }
  async function openRunDirectory() {
    if (completed) await perform(async () => invoke("open_run_directory", { repository, runId: completed.request.run_id }));
  }

  async function saveCodePlan(execute: boolean) {
    await perform(async () => {
      if (!planPath) throw new Error("保存先がありません．既存Planを開き直してください．");
      await invoke<string>("save_analysis_plan_yaml", { repository, path: planPath, yaml: yamlPreview });
      setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: planPath }));
      if (execute) setCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: planPath }));
    });
  }

  const base = sources.find((source) => source.id === baseSource);
  const editorReady = repository.length > 0 && sources.length > 0;
  const roleNames = Object.keys(editor?.config_schema.properties?.roles?.properties ?? {});

  return <main>
    <header><p className="eyebrow">DAVIS MODEL</p><h1>ローカルデータから推定まで</h1>
      <p className="lead">ComponentManifestに従い，データ結合とモデル設定を同じAnalysisPlanとして編集します．</p></header>
    {error && <div className="error sticky-error">{error}</div>}

    <section><SectionHeading number="1" title="Workspace" description="componentとrun記録を置くDavis repositoryを選択します．" />
      <PathField value={repository} placeholder="Davis repository" onChange={(value) => { setRepository(value); setEditor(undefined); }} onChoose={chooseRepository} /></section>

    <section>
      <div className="heading-with-actions"><SectionHeading number="2" title="Analysis plan editor" description="新規作成と既存Planの再編集を同じ画面で行います．" />
        <div className="top-actions"><button className="secondary" onClick={newPlan}>新規Plan</button><button className="secondary" disabled={!repository} onClick={openPlanForEditing}>既存Planを開く</button></div></div>
      <div className="field-grid compact-grid"><label><span>Plan name</span><input value={planName} onChange={(event) => setPlanName(event.target.value)} /></label>
        <label><span>ComponentManifest</span><select value={editor ? `${editor.manifest.id}@${editor.manifest.version}` : ""} disabled={!editorOptions.length || codeMode} onChange={(event) => selectEditor(event.target.value)}>
          {!editor && <option value="">Workspaceを選択してください</option>}{editorOptions.map((item) => <option key={`${item.manifest.id}@${item.manifest.version}`} value={`${item.manifest.id}@${item.manifest.version}`}>{item.manifest.name} ({item.manifest.id} {item.manifest.version})</option>)}</select></label></div>

      {codeMode && <div className="code-mode"><div className="notice">このcomponentはForm editorの対応範囲外です．内容を失わないYAML modeで開いています．</div>
        <textarea className="yaml-preview editable" value={yamlPreview} onChange={(event) => setYamlPreview(event.target.value)} aria-label="model.yaml code editor" />
        <div className="actions"><button className="secondary" disabled={busy} onClick={() => saveCodePlan(false)}>上書き保存・検証</button><button disabled={busy} onClick={() => saveCodePlan(true)}>上書きして実行</button></div></div>}

      {!codeMode && <>
      <div className="subsection-heading"><div><h3>入力データ</h3><p>inspection結果は各データカード内で確認できます．</p></div><button className="secondary" onClick={addDataSources}>CSVを追加</button></div>
      {!sources.length && <div className="empty-state">CSVを追加するか，既存Planを開いてください．</div>}
      {sources.map((source) => <article className="source-card" key={source.id}>
        <div className="source-title"><div><strong>{source.id}</strong><small>{source.path}</small></div><div>
          <button className="text-button" onClick={() => setExpandedSources((current) => ({ ...current, [source.id]: !current[source.id] }))}>{expandedSources[source.id] ? "詳細を閉じる" : "データを確認"}</button>
          <button className="text-button danger-text" onClick={() => removeSource(source.id)}>削除</button></div></div>
        <div className="profile-summary"><span>{source.profile.encoding}</span><span>{source.profile.rows_sampled}行確認</span><span>{source.profile.columns.length}列</span>{source.profile.truncated && <span>sample</span>}</div>
        {expandedSources[source.id] && <ProfileTable profile={source.profile} compact />}
      </article>)}

      {!!sources.length && <>
        <div className="subsection-heading"><div><h3>結合</h3><p>追加表を基準表へ結合します．</p></div></div>
        <label className="wide-label"><span>基準データ</span><select value={baseSource} onChange={(event) => setBaseSource(event.target.value)}>{sources.map((source) => <option key={source.id} value={source.id}>{source.id}</option>)}</select></label>
        {base && sources.filter((source) => source.id !== baseSource).map((source) => {
          const join = joinFor(source.id);
          return <div className="join-row" key={source.id}><strong>{baseSource}</strong>
            <select value={join.leftOn} onChange={(event) => updateJoin(source.id, { leftOn: event.target.value })}><option value="">左キー</option>{base.profile.columns.map((column) => <option key={column.name}>{column.name}</option>)}</select>
            <span className="join-mark">=</span><strong>{source.id}</strong>
            <select value={join.rightOn} onChange={(event) => updateJoin(source.id, { rightOn: event.target.value })}><option value="">右キー</option>{source.profile.columns.map((column) => <option key={column.name}>{column.name}</option>)}</select>
            <select value={join.relationship} onChange={(event) => updateJoin(source.id, { relationship: event.target.value as JoinDraft["relationship"] })}><option value="many_to_one">many to one</option><option value="one_to_one">one to one</option></select>
            <select value={join.how} onChange={(event) => updateJoin(source.id, { how: event.target.value as JoinDraft["how"] })}><option value="left">left</option><option value="inner">inner</option></select>
            <label className="check-label"><input type="checkbox" checked={join.allowUnmatched} onChange={(event) => updateJoin(source.id, { allowUnmatched: event.target.checked })} />未照合を許可</label></div>;
        })}

        <div className="subsection-heading"><div><h3>役割列</h3><p>項目と必須性はcomponentのconfig schema，表示名はUI schemaから読み込みます．</p></div></div>
        <div className="role-grid">{roleNames.map((role) => <ColumnPicker key={role} label={roleLabel(role, editor)} value={roles[role]} sources={sources}
          optional={!(editor?.config_schema.properties?.roles?.required ?? []).includes(role)} onChange={(ref) => selectRole(role, ref)} />)}</div>

        <div className="subsection-heading"><div><h3>効用term</h3><p>選択肢ID列の値から対象候補を生成します．未選択なら全選択肢へ適用します．</p></div><button className="secondary" onClick={addTerm}>termを追加</button></div>
        {!terms.length && <div className="empty-state">説明変数または定数項を追加してください．</div>}
        {terms.map((term) => <div className="term-row" key={term.id}>
          <input aria-label="parameter" value={term.parameter} onChange={(event) => updateTerm(term.id, { parameter: event.target.value })} placeholder="beta_time" />
          <select value={term.mode} onChange={(event) => updateTerm(term.id, { mode: event.target.value as TermDraft["mode"] })}><option value="column">列</option><option value="constant">定数</option></select>
          {term.mode === "column" ? <ColumnPicker value={{ source: term.source, column: term.column }} sources={sources} onChange={(ref) => updateTerm(term.id, ref ?? { source: "", column: "" })} inline />
            : <input type="number" value={term.constant} onChange={(event) => updateTerm(term.id, { constant: event.target.value })} placeholder="1" />}
          <AlternativePicker candidates={alternativeValues?.values ?? []} value={term.alternatives} onChange={(alternatives) => updateTerm(term.id, { alternatives })} />
          <button className="text-button danger-text" onClick={() => setTerms((current) => current.filter((item) => item.id !== term.id))}>削除</button></div>)}
        {alternativeValues && <p className="hint">選択肢候補: {alternativeValues.values.length}件，{alternativeValues.rows_sampled}行から取得{alternativeValues.truncated ? " (上限付きsample)" : ""}</p>}

        <div className="actions editor-actions"><button className="secondary" disabled={!editorReady || busy} onClick={previewPlan}>YAMLを確認</button>
          <button className="secondary" disabled={!editorReady || busy} onClick={() => saveDraft(false)}>別名で保存</button>
          <button className="secondary" disabled={!editorReady || !planPath || busy} onClick={() => saveDraft(false, true)}>上書き保存</button>
          <button disabled={!editorReady || busy} onClick={() => saveDraft(true, Boolean(planPath))}>{planPath ? "上書きして推定" : "保存して推定"}</button></div>
        {validation && <div className="success">{validation.component.id} {validation.component.version}として保存・検証しました．</div>}
        {planPath && <div className="plan-path">{planPath}</div>}
        {yamlPreview && <textarea className="yaml-preview" readOnly value={yamlPreview} aria-label="生成されたmodel.yaml" />}
      </>}
      </>}
    </section>

    {completed && <section><SectionHeading number="3" title="Run result" description={completed.request.run_id} />
      <div className="run-directory-row"><div className="run-directory">{completed.run_directory}</div><button className="secondary" disabled={busy} onClick={openRunDirectory}>結果フォルダを開く</button></div>
      <div className="artifacts">{[...Object.entries(completed.result.artifacts), ...Object.entries(completed.result.extensions)].map(([name, artifact]) =>
        <article key={name}><strong>{name}</strong><span>{artifact.path}</span><small>{artifact.media_type}{artifact.size ? ` · ${artifact.size} bytes` : ""}</small></article>)}</div></section>}
    {busy && <div className="busy">処理中です…</div>}
  </main>;
}

function SectionHeading({ number, title, description }: { number: string; title: string; description: string }) {
  return <div className="section-heading"><span>{number}</span><div><h2>{title}</h2><p>{description}</p></div></div>;
}
function roleLabel(role: string, editor?: ComponentEditor) {
  return editor?.ui_schema.roles?.["ui:labels"]?.[role] ?? defaultRoleLabels[role] ?? role;
}
function sourceId(path: string) {
  const raw = path.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? "data";
  return raw.replace(/[^A-Za-z0-9_]+/g, "_").replace(/^\d/, "data_$&") || "data";
}
function PathField({ value, placeholder, onChange, onChoose }: { value: string; placeholder: string; onChange: (value: string) => void; onChoose: () => void }) {
  return <div className="path-field"><input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} /><button className="secondary" onClick={onChoose}>選択</button></div>;
}
function ColumnPicker({ label, value, sources, optional = false, onChange, inline = false }: {
  label?: string; value?: ColumnRef; sources: SourceDraft[]; optional?: boolean; onChange: (value: ColumnRef | undefined) => void; inline?: boolean;
}) {
  const selectedSource = sources.find((source) => source.id === value?.source);
  const picker = <><select value={value?.source ?? ""} onChange={(event) => {
    const source = sources.find((item) => item.id === event.target.value);
    onChange(source ? { source: source.id, column: source.profile.columns[0]?.name ?? "" } : undefined);
  }}><option value="">{optional ? "使用しない" : "データを選択"}</option>{sources.map((source) => <option key={source.id} value={source.id}>{source.id}</option>)}</select>
    <select value={value?.column ?? ""} disabled={!selectedSource} onChange={(event) => onChange(selectedSource ? { source: selectedSource.id, column: event.target.value } : undefined)}>
      <option value="">列を選択</option>{selectedSource?.profile.columns.map((column) => <option key={column.name} value={column.name}>{column.name} ({column.inferred_type})</option>)}</select></>;
  if (inline) return <div className="inline-column-picker">{picker}</div>;
  return <label><span>{label}</span><div className="inline-column-picker">{picker}</div></label>;
}
function AlternativePicker({ candidates, value, onChange }: { candidates: string[]; value: string; onChange: (value: string) => void }) {
  const [search, setSearch] = useState("");
  const selected = value.split(",").map((item) => item.trim()).filter(Boolean);
  const filtered = candidates.filter((candidate) => candidate.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  const visible = search ? filtered : filtered.slice(0, 8);
  function toggle(candidate: string) {
    onChange((selected.includes(candidate) ? selected.filter((item) => item !== candidate) : [...selected, candidate]).join(", "));
  }
  return <div className="alternative-picker">
    <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={candidates.length ? "選択肢を検索" : "先に選択肢ID列を指定"} disabled={!candidates.length} />
    {!!selected.length && <div className="selected-alternatives">{selected.map((item) => <button type="button" key={item} onClick={() => toggle(item)}>{item} ×</button>)}</div>}
    {!!visible.length && <div className="alternative-menu">{visible.map((candidate) => <button type="button" className={selected.includes(candidate) ? "selected" : ""} key={candidate} onClick={() => toggle(candidate)}>{candidate}</button>)}</div>}
  </div>;
}
function ProfileTable({ profile, compact = false }: { profile: CsvProfile; compact?: boolean }) {
  return <div className={compact ? "profile embedded-profile" : "profile"}>{!compact && <div className="profile-summary"><span>{profile.encoding}</span><span>{JSON.stringify(profile.delimiter)}区切り</span><span>{profile.rows_sampled}行確認</span></div>}
    <table><thead><tr><th>列</th><th>推論型</th><th>欠損</th><th>異なる値</th><th>警告</th></tr></thead><tbody>{profile.columns.map((column) =>
      <tr key={column.name}><td>{column.name}</td><td>{column.inferred_type}</td><td>{column.null_count}</td><td>{column.unique_sample}</td><td>{column.warnings.join(", ") || "—"}</td></tr>)}</tbody></table></div>;
}
