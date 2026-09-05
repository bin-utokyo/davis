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
type SourceDraft = { id: string; path: string; profile: CsvProfile };
type ColumnRef = { source: string; column: string };
type RoleName = "case_id" | "alternative_id" | "chosen" | "available";
type JoinDraft = {
  leftOn: string; rightOn: string; relationship: "many_to_one" | "one_to_one";
  how: "left" | "inner"; allowUnmatched: boolean;
};
type TermDraft = {
  id: number; parameter: string; mode: "column" | "constant"; source: string;
  column: string; constant: string; alternatives: string;
};

const emptyRoles: Record<RoleName, ColumnRef | undefined> = {
  case_id: undefined, alternative_id: undefined, chosen: undefined, available: undefined,
};
const roleLabels: Record<RoleName, string> = {
  case_id: "ケースID", alternative_id: "選択肢ID", chosen: "選択結果", available: "利用可能性 (任意)",
};

export default function App() {
  const [repository, setRepository] = useState("");
  const [csvPath, setCsvPath] = useState("");
  const [planPath, setPlanPath] = useState("");
  const [profile, setProfile] = useState<CsvProfile>();
  const [validation, setValidation] = useState<Validation>();
  const [completed, setCompleted] = useState<CompletedRun>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [planName, setPlanName] = useState("mode-choice-model");
  const [sources, setSources] = useState<SourceDraft[]>([]);
  const [baseSource, setBaseSource] = useState("");
  const [joins, setJoins] = useState<Record<string, JoinDraft>>({});
  const [roles, setRoles] = useState(emptyRoles);
  const [terms, setTerms] = useState<TermDraft[]>([]);
  const [nextTermId, setNextTermId] = useState(1);
  const [yamlPreview, setYamlPreview] = useState("");

  async function perform(action: () => Promise<void>) {
    setBusy(true); setError("");
    try { await action(); } catch (reason) { setError(String(reason)); } finally { setBusy(false); }
  }

  async function chooseRepository() {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") setRepository(selected);
  }

  async function chooseCsv() {
    const selected = await open({ multiple: false, filters: [{ name: "CSV", extensions: ["csv", "tsv"] }] });
    if (typeof selected !== "string") return;
    setCsvPath(selected);
    await perform(async () => setProfile(await invoke<CsvProfile>("inspect_csv_file", { path: selected })));
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
        return { id, path: newProfile.path, profile: newProfile };
      });
      const updated = [...sources, ...added];
      setSources(updated);
      if (!baseSource && updated[0]) { setBaseSource(updated[0].id); suggestRoles(updated[0]); }
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

  function buildPlan(): Record<string, unknown> {
    if (!sources.length) throw new Error("少なくとも1つのCSVを追加してください．");
    if (!baseSource) throw new Error("基準データを選択してください．");
    if (!roles.case_id || !roles.alternative_id || !roles.chosen) throw new Error("ケースID，選択肢ID，選択結果の列を選択してください．");
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
    const sourceMap: Record<string, Record<string, string>> = Object.fromEntries(sources.map((source) => [source.id, { kind: "local", path: source.path }]));
    let choiceData: Record<string, unknown> = sourceMap[sources[0].id];
    if (sources.length > 1) {
      const joinList = sources.filter((source) => source.id !== baseSource).map((source) => {
        const join = joinFor(source.id);
        if (!join.leftOn || !join.rightOn) throw new Error(`${source.id}の結合キーを選択してください．`);
        return { source: source.id, left_on: join.leftOn, right_on: join.rightOn, relationship: join.relationship, how: join.how, allow_unmatched: join.allowUnmatched };
      });
      choiceData = {
        kind: "table_binding", processor: { id: "davis/csv-transform", version: "0.4.0" },
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
      component: { id: "davis/mnl", version: "0.2.0", operation: "estimate" },
      inputs: { choice_data: choiceData },
      config: { roles: configRoles, terms: configTerms, estimation: { optimizer: "bfgs", max_iterations: 500, tolerance: 1e-8 } },
      run: { label: name, tags: ["gui", "mnl"] },
    };
  }

  async function previewPlan() {
    await perform(async () => setYamlPreview(await invoke<string>("render_analysis_plan", { plan: buildPlan() })));
  }
  async function saveDraft(execute: boolean) {
    await perform(async () => {
      const target = await save({ defaultPath: repository ? `${repository}/model.yaml` : "model.yaml", filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] });
      if (!target) return;
      const plan = buildPlan();
      const saved = await invoke<string>("save_analysis_plan", { path: target, plan });
      setPlanPath(saved); setYamlPreview(await invoke<string>("render_analysis_plan", { plan }));
      setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: saved }));
      setCompleted(undefined);
      if (execute) setCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: saved }));
    });
  }
  async function choosePlan() {
    const selected = await open({ multiple: false, filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }] });
    if (typeof selected === "string") setPlanPath(selected);
  }
  async function validate() {
    await perform(async () => { setValidation(await invoke<Validation>("validate_analysis_plan", { repository, plan: planPath })); setCompleted(undefined); });
  }
  async function run() {
    await perform(async () => setCompleted(await invoke<CompletedRun>("run_analysis_plan", { repository, plan: planPath })));
  }
  async function openRunDirectory() {
    if (completed) await perform(async () => invoke("open_run_directory", { repository, runId: completed.request.run_id }));
  }

  const base = sources.find((source) => source.id === baseSource);
  const ready = repository.length > 0 && planPath.length > 0;
  const editorReady = repository.length > 0 && sources.length > 0;

  return <main>
    <header><p className="eyebrow">DAVIS MODEL</p><h1>ローカルデータから推定まで</h1>
      <p className="lead">複数のCSVを結合し，MNLの構造を画面で定義して，共通のmodel.yamlとして保存・実行します．</p></header>
    {error && <div className="error sticky-error">{error}</div>}

    <section><SectionHeading number="1" title="Workspace" description="componentとrun記録を置くDavis repositoryを選択します．" />
      <PathField value={repository} placeholder="Davis repository" onChange={setRepository} onChoose={chooseRepository} /></section>

    <section><SectionHeading number="2" title="MNL plan editor" description="入力，結合，役割列，効用termからmodel.yamlを作成します．" />
      <div className="field-grid compact-grid"><label><span>Plan name</span><input value={planName} onChange={(event) => setPlanName(event.target.value)} /></label>
        <label><span>Model</span><input value="davis/mnl 0.2.0" disabled /></label></div>
      <div className="subsection-heading"><div><h3>入力データ</h3><p>基準表と，結合する追加表を登録します．</p></div><button className="secondary" onClick={addDataSources}>CSVを追加</button></div>
      {!sources.length && <div className="empty-state">CSVを追加すると，列と結合設定がここに表示されます．</div>}
      {sources.map((source) => <article className="source-card" key={source.id}><div className="source-title"><div><strong>{source.id}</strong><small>{source.path}</small></div>
        <button className="text-button danger-text" onClick={() => removeSource(source.id)}>削除</button></div>
        <div className="profile-summary"><span>{source.profile.encoding}</span><span>{source.profile.rows_sampled}行確認</span><span>{source.profile.columns.length}列</span></div></article>)}

      {!!sources.length && <>
        <div className="subsection-heading"><div><h3>結合</h3><p>追加表は現在，基準表へ直接結合します．</p></div></div>
        <label className="wide-label"><span>基準データ</span><select value={baseSource} onChange={(event) => setBaseSource(event.target.value)}>
          {sources.map((source) => <option key={source.id} value={source.id}>{source.id}</option>)}</select></label>
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

        <div className="subsection-heading"><div><h3>役割列</h3><p>long形式の選択データが持つ意味を指定します．</p></div></div>
        <div className="role-grid">{(Object.keys(roleLabels) as RoleName[]).map((role) => <ColumnPicker key={role} label={roleLabels[role]} value={roles[role]} sources={sources}
          optional={role === "available"} onChange={(ref) => setRoles((current) => ({ ...current, [role]: ref }))} />)}</div>

        <div className="subsection-heading"><div><h3>効用term</h3><p>適用する選択肢はカンマ区切りです．空欄なら全選択肢へ適用します．</p></div><button className="secondary" onClick={addTerm}>termを追加</button></div>
        {!terms.length && <div className="empty-state">説明変数または定数項を追加してください．</div>}
        {terms.map((term) => <div className="term-row" key={term.id}>
          <input aria-label="parameter" value={term.parameter} onChange={(event) => updateTerm(term.id, { parameter: event.target.value })} placeholder="beta_time" />
          <select value={term.mode} onChange={(event) => updateTerm(term.id, { mode: event.target.value as TermDraft["mode"] })}><option value="column">列</option><option value="constant">定数</option></select>
          {term.mode === "column" ? <ColumnPicker value={{ source: term.source, column: term.column }} sources={sources} onChange={(ref) => updateTerm(term.id, ref ?? { source: "", column: "" })} inline />
            : <input type="number" value={term.constant} onChange={(event) => updateTerm(term.id, { constant: event.target.value })} placeholder="1" />}
          <input value={term.alternatives} onChange={(event) => updateTerm(term.id, { alternatives: event.target.value })} placeholder="train, car (任意)" />
          <button className="text-button danger-text" onClick={() => setTerms((current) => current.filter((item) => item.id !== term.id))}>削除</button></div>)}
        <div className="actions editor-actions"><button className="secondary" disabled={!editorReady || busy} onClick={previewPlan}>YAMLを確認</button>
          <button className="secondary" disabled={!editorReady || busy} onClick={() => saveDraft(false)}>保存して検証</button><button disabled={!editorReady || busy} onClick={() => saveDraft(true)}>保存して推定</button></div>
        {yamlPreview && <textarea className="yaml-preview" readOnly value={yamlPreview} aria-label="生成されたmodel.yaml" />}
      </>}
    </section>

    <section><SectionHeading number="3" title="Existing analysis plan" description="手書きまたはAIが作成したmodel.yamlも同じ経路で検証・実行できます．" />
      <PathField value={planPath} placeholder="model.yaml" onChange={setPlanPath} onChoose={choosePlan} /><div className="actions">
        <button className="secondary" disabled={!ready || busy} onClick={validate}>検証する</button><button disabled={!ready || busy} onClick={run}>推定を実行する</button></div>
      {validation && <div className="success">{validation.component.id} {validation.component.version}を実行できます．</div>}</section>

    <section><SectionHeading number="4" title="CSV inspection" description="単独のCSVについてencoding，型推論，先頭0を確認します．" />
      <PathField value={csvPath} placeholder="CSV file" onChange={setCsvPath} onChoose={chooseCsv} />{profile && <ProfileTable profile={profile} />}</section>

    {completed && <section><SectionHeading number="5" title="Run result" description={completed.request.run_id} />
      <div className="run-directory-row"><div className="run-directory">{completed.run_directory}</div><button className="secondary" disabled={busy} onClick={openRunDirectory}>結果フォルダを開く</button></div>
      <div className="artifacts">{[...Object.entries(completed.result.artifacts), ...Object.entries(completed.result.extensions)].map(([name, artifact]) =>
        <article key={name}><strong>{name}</strong><span>{artifact.path}</span><small>{artifact.media_type}{artifact.size ? ` · ${artifact.size} bytes` : ""}</small></article>)}</div></section>}
    {busy && <div className="busy">処理中です…</div>}
  </main>;
}

function SectionHeading({ number, title, description }: { number: string; title: string; description: string }) {
  return <div className="section-heading"><span>{number}</span><div><h2>{title}</h2><p>{description}</p></div></div>;
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
function ProfileTable({ profile }: { profile: CsvProfile }) {
  return <div className="profile"><div className="profile-summary"><span>{profile.encoding}</span><span>{JSON.stringify(profile.delimiter)}区切り</span><span>{profile.rows_sampled}行確認</span></div>
    <table><thead><tr><th>列</th><th>推論型</th><th>欠損</th><th>異なる値</th><th>警告</th></tr></thead><tbody>{profile.columns.map((column) =>
      <tr key={column.name}><td>{column.name}</td><td>{column.inferred_type}</td><td>{column.null_count}</td><td>{column.unique_sample}</td><td>{column.warnings.join(", ") || "—"}</td></tr>)}</tbody></table></div>;
}
