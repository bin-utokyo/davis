import { invoke } from "@tauri-apps/api/core";
import { useEffect, useRef, useState } from "react";

type ColumnProfile = { name: string; inferred_type: string };
type CsvProfile = { path: string; encoding: string; rows_sampled: number; truncated: boolean; columns: ColumnProfile[] };
export type FormSource = {
  id: string; path: string; serializedPath: string; read?: unknown; profile: CsvProfile;
  origin?: { kind: "local" } | { kind: "catalog"; dataset_id: string; file_id: string; revision?: string };
};
export type FormJoin = { leftOn: string; rightOn: string; relationship: "many_to_one" | "one_to_one"; how: "left" | "inner"; allowUnmatched: boolean };
export type ColumnBinding = { source: string; column: string };
export type FormInput = { sources: FormSource[]; base: string; joins: Record<string, FormJoin>; columns?: Record<string, ColumnBinding>; processor?: { id: string; version: string }; forceBinding?: boolean };
export type JsonSchema = {
  type?: string | string[]; enum?: unknown[]; default?: unknown; required?: string[]; title?: string; description?: string;
  properties?: Record<string, JsonSchema>; items?: JsonSchema; additionalProperties?: boolean | JsonSchema; oneOf?: JsonSchema[];
};
type FormSection = {
  bind: string; widget?: string; input?: string; title?: string; description?: string; labels?: Record<string, string>;
  allow_constant?: boolean; show_coefficient?: boolean; alternatives_from?: string; parameters_from?: string; context?: Record<string, ContextProvider>;
};
type ContextProvider = { provider: "config" | "columns" | "distinct-values"; path?: string; input?: string; column_from?: string };
type InputPresentation = { title?: string; description?: string; widget?: string; preparation?: { component: string; version: string } };
export type FormDefinition = {
  version?: string; inputs?: Record<string, InputPresentation>; sections?: FormSection[]; defaults?: Record<string, unknown>;
  results?: Array<{ artifact: string; title: string; widget: "key-value" | "table" }>;
};
type UiExtension = { api_version: string; html: string };
type EditorDefinition = { config_schema: JsonSchema; ui_schema: FormDefinition; ui_extensions?: Record<string, UiExtension> };
type DistinctValues = { values: string[]; rows_sampled: number; truncated: boolean };

export function SchemaFormEditor({ definition, inputs, config, onAddSources, onAddCatalog, onInputChange, onConfigChange }: {
  definition: EditorDefinition; inputs: Record<string, FormInput | undefined>; config: Record<string, unknown>;
  onAddSources: (slot: string) => void; onAddCatalog: (slot: string) => void; onInputChange: (slot: string, input: FormInput | undefined) => void;
  onConfigChange: (config: Record<string, unknown>) => void;
}) {
  const form = definition.ui_schema; const sections = form.sections ?? []; const [alternatives, setAlternatives] = useState<DistinctValues>();
  const alternativePath = sections.find((section) => section.alternatives_from)?.alternatives_from;
  const alternativeOwner = alternativePath?.split("/").slice(0, -1).join("/");
  const alternativeAlias = alternativePath ? getAt(config, alternativePath) : undefined;
  const alternativeSection = sections.find((section) => section.bind === alternativeOwner);
  const alternativeInput = alternativeSection?.input ? inputs[alternativeSection.input] : undefined;
  const alternativeBinding = alternativeInput && typeof alternativeAlias === "string" ? bindingColumns(alternativeInput).find((candidate) => candidate.alias === alternativeAlias) : undefined;
  const alternativeSource = alternativeInput?.sources.find((source) => source.id === alternativeBinding?.source);

  useEffect(() => {
    let active = true;
    if (alternativeSource && alternativeBinding) {
      invoke<DistinctValues>("inspect_distinct_values", { path: alternativeSource.path, column: alternativeBinding.column })
        .then((value) => { if (active) setAlternatives(value); }).catch(() => { if (active) setAlternatives(undefined); });
    } else setAlternatives(undefined);
    return () => { active = false; };
  }, [alternativeSource?.path, alternativeBinding?.column]);

  function update(path: string, value: unknown) { onConfigChange(setAt(config, path, value)); }
  return <>
    <div className="subsection-heading"><div><h3>入力データ</h3><p>入力と結合はモデルではなくDavis共通のtable bindingとして保存します．</p></div></div>
    {Object.entries(form.inputs ?? {}).map(([slot, metadata]) => <InputBindingEditor key={slot} slot={slot} metadata={metadata} input={inputs[slot]}
      onAdd={() => onAddSources(slot)} onAddCatalog={() => onAddCatalog(slot)} onChange={(input) => onInputChange(slot, input)} />)}
    {sections.map((section) => {
      const schema = schemaAt(definition.config_schema, section.bind); const value = getAt(config, section.bind);
      const input = section.input ? inputs[section.input] : undefined; const widget = section.widget ?? "auto";
      const extensionId = widget.startsWith("extension:") ? widget.slice("extension:".length) : undefined;
      const extension = extensionId ? definition.ui_extensions?.[extensionId] : undefined;
      return <div className="manifest-section" key={section.bind}>
        {widget === "column-map" && <ColumnMap section={section} schema={schema} value={asObject(value)} input={input} onChange={(next) => update(section.bind, next)} />}
        {widget === "utility-terms" && <UtilityTerms section={section} value={asArray(value)} input={input} candidates={alternatives?.values ?? []} onChange={(next) => update(section.bind, next)} />}
        {widget === "nests" && <NestEditor section={section} value={asArray(value)} candidates={alternatives?.values ?? []} onChange={(next) => update(section.bind, next)} />}
        {widget === "parameter-settings" && <ParameterSettings section={section} schema={schema} value={asObject(value)} config={config} onChange={(next) => update(section.bind, next)} />}
        {(widget === "auto" || widget === "object") && <AutoSection section={section} schema={schema} value={value} onChange={(next) => update(section.bind, next)} />}
        {extension && <ExtensionSection section={section} extension={extension} schema={schema} value={value} config={config} inputs={inputs} onChange={(next) => update(section.bind, next)} />}
        {!knownWidget(widget) && !extension && <YamlFallback section={section} value={value} onChange={(next) => update(section.bind, next)} reason={extensionId ? `UI extension ${extensionId}を読み込めません．` : `widget ${widget}はこのDesktopに未実装です．`} />}
      </div>;
    })}
    {alternatives && <p className="hint">候補値: {alternatives.values.length}件，{alternatives.rows_sampled}行から取得{alternatives.truncated ? " (上限付きsample)" : ""}</p>}
  </>;
}

function ExtensionSection({ section, extension, schema, value, config, inputs, onChange }: { section: FormSection; extension: UiExtension; schema?: JsonSchema; value: unknown; config: Record<string, unknown>; inputs: Record<string, FormInput | undefined>; onChange: (value: unknown) => void }) {
  const [context, setContext] = useState<Record<string, unknown>>({}); const [contextErrors, setContextErrors] = useState<Record<string, string>>({});
  useEffect(() => {
    let active = true;
    Promise.all(Object.entries(section.context ?? {}).map(async ([name, declaration]) => {
      try { return [name, await resolveContext(declaration, config, inputs), undefined] as const; }
      catch (reason) { return [name, undefined, String(reason)] as const; }
    })).then((entries) => {
      if (!active) return;
      const resolvedContext: Record<string, unknown> = {}; const errors: Record<string, string> = {};
      for (const [name, resolved, error] of entries) { if (error) errors[name] = error; else resolvedContext[name] = resolved; }
      setContext(resolvedContext); setContextErrors(errors);
    });
    return () => { active = false; };
  }, [section.context, config, inputs]);
  return <ExtensionWidget section={section} extension={extension} schema={schema} value={value} context={context} contextErrors={contextErrors} onChange={onChange} />;
}

async function resolveContext(declaration: ContextProvider, config: Record<string, unknown>, inputs: Record<string, FormInput | undefined>): Promise<unknown> {
  if (declaration.provider === "config") return getAt(config, declaration.path ?? "");
  const input = declaration.input ? inputs[declaration.input] : undefined;
  if (!input) throw new Error(`入力${declaration.input ?? ""}が未選択です．`);
  const columns = bindingColumns(input);
  if (declaration.provider === "columns") return columns;
  const alias = getAt(config, declaration.column_from ?? "");
  if (typeof alias !== "string" || !alias) throw new Error(`列参照${declaration.column_from ?? ""}が未設定です．`);
  const binding = columns.find((candidate) => candidate.alias === alias);
  const source = input.sources.find((candidate) => candidate.id === binding?.source);
  if (!binding || !source) throw new Error(`列${alias}を入力${declaration.input ?? ""}から解決できません．`);
  return invoke<DistinctValues>("inspect_distinct_values", { path: source.path, column: binding.column });
}

function ExtensionWidget({ section, extension, schema, value, context, contextErrors, onChange }: { section: FormSection; extension: UiExtension; schema?: JsonSchema; value: unknown; context: Record<string, unknown>; contextErrors: Record<string, string>; onChange: (value: unknown) => void }) {
  const frame = useRef<HTMLIFrameElement>(null); const [height, setHeight] = useState(180);
  const document = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none';"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${extension.html}</body></html>`;
  function render() { frame.current?.contentWindow?.postMessage({ source: "davis-host", api_version: "davis.widget/v1", type: "render", payload: { value, schema, context, context_errors: contextErrors, section } }, "*"); }
  useEffect(() => {
    function receive(event: MessageEvent) {
      if (event.source !== frame.current?.contentWindow || !event.data || typeof event.data !== "object" || event.data.source !== "davis-widget" || event.data.api_version !== "davis.widget/v1") return;
      if (event.data.type === "ready") render();
      if (event.data.type === "set-value") onChange(event.data.value);
      if (event.data.type === "resize" && Number.isFinite(event.data.height)) setHeight(Math.min(900, Math.max(120, Number(event.data.height))));
    }
    window.addEventListener("message", receive); return () => window.removeEventListener("message", receive);
  });
  useEffect(render, [value, schema, context, contextErrors, section]);
  return <><SectionTitle section={section} /><iframe className="ui-extension" ref={frame} title={section.title ?? section.bind} sandbox="allow-scripts" srcDoc={document} style={{ height }} onLoad={render} /></>;
}

function InputBindingEditor({ slot, metadata, input, onAdd, onAddCatalog, onChange }: { slot: string; metadata: InputPresentation; input?: FormInput; onAdd: () => void; onAddCatalog: () => void; onChange: (input: FormInput | undefined) => void }) {
  const sources = input?.sources ?? []; const base = sources.find((source) => source.id === input?.base) ?? sources[0];
  function remove(id: string) { if (!input) return; const remaining = input.sources.filter((source) => source.id !== id); if (!remaining.length) return onChange(undefined); const joins = { ...input.joins }; delete joins[id]; onChange({ ...input, sources: remaining, base: input.base === id ? remaining[0].id : input.base, joins }); }
  function updateJoin(id: string, patch: Partial<FormJoin>) { if (input) onChange({ ...input, joins: { ...input.joins, [id]: { ...defaultJoin(), ...input.joins[id], ...patch } } }); }
  return <article className="binding-editor"><div className="binding-heading"><div><strong>{metadata.title ?? slot}</strong><code>{slot}</code>{metadata.description && <p>{metadata.description}</p>}</div><div className="source-actions"><button className="secondary" onClick={onAdd}>ローカルCSV</button><button className="secondary" onClick={onAddCatalog}>Davis Catalog</button></div></div>
    {!sources.length && <div className="empty-state compact-empty">CSVを1つ以上選択してください．</div>}
    {sources.map((source) => <div className="binding-source" key={source.id}><div><strong>{source.id}</strong><small>{source.origin?.kind === "catalog" ? `${source.origin.dataset_id} / ${source.origin.file_id}` : source.path}</small></div><div className="profile-summary">{source.origin?.kind === "catalog" && <span>Catalog</span>}<span>{source.profile.encoding}</span><span>{source.profile.rows_sampled}行</span><span>{source.profile.columns.length}列</span></div><button className="text-button danger-text" onClick={() => remove(source.id)}>削除</button></div>)}
    {sources.length > 1 && input && <><label className="wide-label"><span>基準データ</span><select value={input.base} onChange={(event) => onChange({ ...input, base: event.target.value })}>{sources.map((source) => <option key={source.id}>{source.id}</option>)}</select></label>
      {base && sources.filter((source) => source.id !== input.base).map((source) => { const join = { ...defaultJoin(), ...input.joins[source.id] }; return <div className="join-row" key={source.id}><strong>{base.id}</strong>
        <select value={join.leftOn} onChange={(event) => updateJoin(source.id, { leftOn: event.target.value })}><option value="">左キー</option>{base.profile.columns.map((column) => <option key={column.name}>{column.name}</option>)}</select><span className="join-mark">=</span><strong>{source.id}</strong>
        <select value={join.rightOn} onChange={(event) => updateJoin(source.id, { rightOn: event.target.value })}><option value="">右キー</option>{source.profile.columns.map((column) => <option key={column.name}>{column.name}</option>)}</select>
        <select value={join.relationship} onChange={(event) => updateJoin(source.id, { relationship: event.target.value as FormJoin["relationship"] })}><option value="many_to_one">many to one</option><option value="one_to_one">one to one</option></select>
        <select value={join.how} onChange={(event) => updateJoin(source.id, { how: event.target.value as FormJoin["how"] })}><option value="left">left</option><option value="inner">inner</option></select><label className="check-label"><input type="checkbox" checked={join.allowUnmatched} onChange={(event) => updateJoin(source.id, { allowUnmatched: event.target.checked })} />未照合を許可</label></div>; })}</>}
  </article>;
}

function SectionTitle({ section }: { section: FormSection }) { return <div className="subsection-heading"><div><h3>{section.title ?? section.bind}</h3><p>{section.description ?? <><code>{section.bind}</code>としてAnalysis Planへ保存します．</>}</p></div></div>; }
function ColumnMap({ section, schema, value, input, onChange }: { section: FormSection; schema?: JsonSchema; value: Record<string, unknown>; input?: FormInput; onChange: (value: Record<string, unknown>) => void }) {
  const names = Object.keys(schema?.properties ?? {}); const required = schema?.required ?? []; const columns = input ? bindingColumns(input) : [];
  return <><SectionTitle section={section} /><div className="role-grid">{names.map((name) => { const property = schema?.properties?.[name]; const acceptsMany = schemaAcceptsArray(property); return <label key={name}><span>{section.labels?.[name] ?? name}{required.includes(name) ? " *" : " (任意)"}</span>{acceptsMany ? <AlternativePicker candidates={columns.map((column) => column.alias)} selected={Array.isArray(value[name]) ? (value[name] as unknown[]).map(String) : typeof value[name] === "string" ? [String(value[name])] : []} onChange={(selected) => onChange(selected.length ? { ...value, [name]: selected } : without(value, name))} /> : <select value={typeof value[name] === "string" ? value[name] as string : ""} disabled={!input} onChange={(event) => onChange(event.target.value ? { ...value, [name]: event.target.value } : without(value, name))}><option value="">列を選択</option>{columns.map((column) => <option key={column.alias} value={column.alias}>{column.label}</option>)}</select>}</label>; })}</div></>;
}
function UtilityTerms({ section, value, input, candidates, onChange }: { section: FormSection; value: unknown[]; input?: FormInput; candidates: string[]; onChange: (value: unknown[]) => void }) {
  const terms = value.map(asObject); const columns = input ? bindingColumns(input) : []; function patch(index: number, change: Record<string, unknown>) { onChange(terms.map((term, current) => current === index ? clean({ ...term, ...change }) : term)); }
  return <><div className="subsection-heading"><div><h3>{section.title ?? section.bind}</h3><p>parameterと効用へ入れる列を指定します．</p></div><button className="secondary" onClick={() => onChange([...terms, { parameter: `beta_${terms.length + 1}`, column: "", ...(section.show_coefficient ? { coefficient: 1 } : {}) }])}>termを追加</button></div>{!terms.length && <div className="empty-state">効用termを追加してください．</div>}
    {terms.map((term, index) => { const usesConstant = "constant" in term; return <div className={`term-row ${section.show_coefficient ? "with-coefficient" : ""}`} key={index}><input aria-label="parameter" value={String(term.parameter ?? "")} onChange={(event) => patch(index, { parameter: event.target.value })} placeholder="beta_time" />
      {section.allow_constant && <select value={usesConstant ? "constant" : "column"} onChange={(event) => patch(index, event.target.value === "constant" ? { constant: 1, column: undefined } : { column: "", constant: undefined })}><option value="column">列</option><option value="constant">定数</option></select>}
      {usesConstant ? <input type="number" value={String(term.constant ?? 1)} onChange={(event) => patch(index, { constant: numberOrEmpty(event.target.value) })} /> : <select value={String(term.column ?? "")} disabled={!input} onChange={(event) => patch(index, { column: event.target.value })}><option value="">列を選択</option>{columns.map((column) => <option key={column.alias} value={column.alias}>{column.label}</option>)}</select>}
      {section.show_coefficient && <input type="number" title="coefficient" value={String(term.coefficient ?? 1)} onChange={(event) => patch(index, { coefficient: numberOrEmpty(event.target.value) })} />}{section.alternatives_from && <AlternativePicker candidates={candidates} selected={(term.alternatives as unknown[] | undefined)?.map(String) ?? []} onChange={(selected) => patch(index, { alternatives: selected.length ? selected : undefined })} />}<button className="text-button danger-text" onClick={() => onChange(terms.filter((_, current) => current !== index))}>削除</button></div>; })}</>;
}
function NestEditor({ section, value, candidates, onChange }: { section: FormSection; value: unknown[]; candidates: string[]; onChange: (value: unknown[]) => void }) {
  const nests = value.map(asObject); function patch(index: number, change: Record<string, unknown>) { onChange(nests.map((nest, current) => current === index ? { ...nest, ...change } : nest)); }
  const estimateLabel = section.labels?.estimate ?? "推定"; const fixedLabel = section.labels?.fixed ?? "固定";
  return <><div className="subsection-heading"><div><h3>{section.title ?? section.bind}</h3><p>{section.description ?? "各選択肢を重複なく1つのnestへ入れます．"}</p></div><button className="secondary" onClick={() => onChange([...nests, { name: `nest_${nests.length + 1}`, alternatives: [], dissimilarity: { initial: 0.8 } }])}>Nestを追加</button></div>{!nests.length && <div className="empty-state">2つ以上のnestを追加してください．</div>}{nests.map((nest, index) => { const dissimilarity = asObject(nest.dissimilarity); const fixed = "fixed" in dissimilarity; const valueLabel = section.labels?.[fixed ? "fixed_value" : "estimate_value"] ?? (fixed ? "固定値" : "推定初期値"); return <div className="nest-row" key={index}><input value={String(nest.name ?? "")} onChange={(event) => patch(index, { name: event.target.value })} placeholder="motorized" /><AlternativePicker candidates={candidates} selected={(nest.alternatives as unknown[] | undefined)?.map(String) ?? []} onChange={(selected) => patch(index, { alternatives: selected })} /><select aria-label={section.labels?.mode ?? "推定または固定"} value={fixed ? "fixed" : "initial"} onChange={(event) => patch(index, { dissimilarity: event.target.value === "fixed" ? { fixed: 1 } : { initial: 0.8 } })}><option value="initial">{estimateLabel}</option><option value="fixed">{fixedLabel}</option></select><input aria-label={valueLabel} title={valueLabel} type="number" min="0.05" max="1" step="0.05" value={String(dissimilarity[fixed ? "fixed" : "initial"] ?? (fixed ? 1 : 0.8))} onChange={(event) => patch(index, { dissimilarity: { [fixed ? "fixed" : "initial"]: numberOrEmpty(event.target.value) } })} /><button className="text-button danger-text" onClick={() => onChange(nests.filter((_, current) => current !== index))}>削除</button></div>; })}</>;
}
function ParameterSettings({ section, schema, value, config, onChange }: { section: FormSection; schema?: JsonSchema; value: Record<string, unknown>; config: Record<string, unknown>; onChange: (value: Record<string, unknown>) => void }) {
  const terms = asArray(getAt(config, section.parameters_from ?? "/terms")).map(asObject); const names = [...new Set(terms.map((term) => String(term.parameter ?? "")).filter(Boolean))]; const itemSchema = typeof schema?.additionalProperties === "object" ? schema.additionalProperties : undefined; const fields = Object.keys(itemSchema?.properties ?? { initial: {}, lower: {}, upper: {} }); function update(name: string, field: string, raw: string) { const current = asObject(value[name]); const next = raw === "" ? without(current, field) : { ...current, [field]: Number(raw) }; onChange({ ...value, [name]: next }); }
  return <><SectionTitle section={section} />{!names.length && <div className="empty-state">先に効用termを追加してください．</div>}{names.map((name) => { const settings = asObject(value[name]); return <div className="parameter-row" key={name}><strong>{name}</strong>{fields.map((field) => <label key={field}><span>{field}</span><input type="number" value={settings[field] === undefined ? "" : String(settings[field])} placeholder="任意" onChange={(event) => update(name, field, event.target.value)} /></label>)}</div>; })}</>;
}
function AutoSection({ section, schema, value, onChange }: { section: FormSection; schema?: JsonSchema; value: unknown; onChange: (value: unknown) => void }) { return <><SectionTitle section={section} />{schema && canAutoRender(schema) ? schema.type === "object" && schema.properties ? <div className="field-grid compact-grid"><AutoFields schema={schema} value={asObject(value)} onChange={onChange} /></div> : <div className="field-grid"><label><span>{schema.title ?? section.title ?? section.bind}</span><ScalarField schema={schema} value={value ?? schema.default} onChange={onChange} /></label></div> : <YamlFallback section={section} value={value} onChange={onChange} reason="自由形式または配列の設定です．" />}</>; }
function AutoFields({ schema, value, onChange }: { schema: JsonSchema; value: Record<string, unknown>; onChange: (value: Record<string, unknown>) => void }) { return <>{Object.entries(schema.properties ?? {}).map(([name, property]) => <label key={name}><span>{property.title ?? name}{schema.required?.includes(name) ? " *" : ""}</span><ScalarField schema={property} value={value[name] ?? property.default} onChange={(next) => onChange(clean({ ...value, [name]: next }))} />{property.description && <small>{property.description}</small>}</label>)}</>; }
function ScalarField({ schema, value, onChange }: { schema: JsonSchema; value: unknown; onChange: (value: unknown) => void }) { if (schema.enum) return <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}><option value="">選択</option>{schema.enum.map((item) => <option key={String(item)} value={String(item)}>{String(item)}</option>)}</select>; if (schema.type === "object" && schema.properties) return <fieldset className="nested-fields"><AutoFields schema={schema} value={asObject(value)} onChange={onChange} /></fieldset>; if (schema.type === "boolean") return <input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} />; if (schema.type === "number" || schema.type === "integer") return <input type="number" step={schema.type === "integer" ? 1 : "any"} value={value === undefined ? "" : String(value)} onChange={(event) => onChange(numberOrEmpty(event.target.value))} />; return <input type="text" value={value === undefined ? "" : String(value)} onChange={(event) => onChange(event.target.value || undefined)} />; }
function YamlFallback({ section, value, onChange, reason }: { section: FormSection; value: unknown; onChange: (value: unknown) => void; reason: string }) {
  const [text, setText] = useState(""); const [error, setError] = useState(""); useEffect(() => { invoke<string>("render_yaml_value", { value }).then(setText).catch((reason) => setError(String(reason))); }, [value]);
  async function apply() { try { onChange(await invoke<unknown>("parse_yaml_value", { yaml: text })); setError(""); } catch (reason) { setError(String(reason)); } }
  return <div className="section-fallback"><p>{reason} このsectionだけYAMLで編集できます．</p><textarea value={text} onChange={(event) => setText(event.target.value)} aria-label={`${section.bind} YAML editor`} /><button className="secondary" onClick={apply}>このsectionへ反映</button>{error && <div className="error">{error}</div>}</div>;
}
function AlternativePicker({ candidates, selected, onChange }: { candidates: string[]; selected: string[]; onChange: (value: string[]) => void }) { const [search, setSearch] = useState(""); const [open, setOpen] = useState(false); const visible = candidates.filter((candidate) => candidate.toLocaleLowerCase().includes(search.toLocaleLowerCase())).slice(0, search ? 50 : 8); function toggle(candidate: string) { onChange(selected.includes(candidate) ? selected.filter((item) => item !== candidate) : [...selected, candidate]); } return <div className="alternative-picker"><div className="alternative-control">{selected.map((item) => <button type="button" className="selected-tag" key={item} onMouseDown={(event) => event.preventDefault()} onClick={() => toggle(item)}>{item} ×</button>)}<input value={search} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onChange={(event) => setSearch(event.target.value)} placeholder={selected.length ? "検索…" : candidates.length ? "値を検索" : "候補列を先に指定"} disabled={!candidates.length} /></div>{open && !!visible.length && <div className="alternative-menu">{visible.map((candidate) => <button type="button" className={selected.includes(candidate) ? "selected" : ""} key={candidate} onMouseDown={(event) => event.preventDefault()} onClick={() => toggle(candidate)}>{candidate}</button>)}</div>}</div>; }

export function bindingColumns(input: FormInput): Array<ColumnBinding & { alias: string; label: string }> {
  const result: Array<ColumnBinding & { alias: string; label: string }> = []; const usedAliases = new Set<string>(); const usedRefs = new Set<string>();
  for (const [alias, ref] of Object.entries(input.columns ?? {})) { if (!input.sources.some((source) => source.id === ref.source && source.profile.columns.some((column) => column.name === ref.column))) continue; result.push({ ...ref, alias, label: `${alias} (${ref.source}.${ref.column})` }); usedAliases.add(alias); usedRefs.add(`${ref.source}\0${ref.column}`); }
  for (const source of input.sources) for (const column of source.profile.columns) { if (usedRefs.has(`${source.id}\0${column.name}`)) continue; let alias = column.name; let suffix = 2; if (usedAliases.has(alias)) alias = `${source.id}_${column.name}`; while (usedAliases.has(alias)) alias = `${source.id}_${column.name}_${suffix++}`; result.push({ source: source.id, column: column.name, alias, label: input.sources.length > 1 ? `${alias} (${source.id}.${column.name})` : `${alias} (${column.inferred_type})` }); usedAliases.add(alias); }
  return result;
}
function defaultJoin(): FormJoin { return { leftOn: "", rightOn: "", relationship: "many_to_one", how: "left", allowUnmatched: false }; }
function knownWidget(widget: string) { return ["column-map", "utility-terms", "nests", "parameter-settings", "auto", "object"].includes(widget); }
function canAutoRender(schema: JsonSchema): boolean { if (schema.type === "array" || Array.isArray(schema.type)) return false; if (schema.type === "object") return Boolean(schema.properties) && Object.values(schema.properties ?? {}).every(canAutoRender); return true; }
function schemaAcceptsArray(schema?: JsonSchema): boolean { return schema?.type === "array" || schema?.oneOf?.some((option) => option.type === "array") === true; }
function pathParts(path: string) { return path.trim().replace(/^\//, "").split("/").filter(Boolean).map((part) => part.replace(/~1/g, "/").replace(/~0/g, "~")); }
function getAt(value: unknown, path: string): unknown { return pathParts(path).reduce<unknown>((current, key) => asObject(current)[key], value); }
function setAt(root: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> { const [head, ...tail] = pathParts(path); return { ...root, [head]: tail.length ? setAt(asObject(root[head]), `/${tail.join("/")}`, value) : value }; }
function schemaAt(schema: JsonSchema, path: string): JsonSchema | undefined { return pathParts(path).reduce<JsonSchema | undefined>((current, key) => current?.properties?.[key], schema); }
function asObject(value: unknown): Record<string, unknown> { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}; }
function asArray(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function without(value: Record<string, unknown>, key: string) { const next = { ...value }; delete next[key]; return next; }
function clean<T extends Record<string, unknown>>(value: T): T { return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as T; }
function numberOrEmpty(value: string) { return value === "" ? undefined : Number(value); }
