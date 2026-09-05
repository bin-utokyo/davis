import { invoke } from "@tauri-apps/api/core";
import { useEffect, useState } from "react";

type ColumnProfile = { name: string; inferred_type: string };
type CsvProfile = { path: string; encoding: string; rows_sampled: number; truncated: boolean; columns: ColumnProfile[] };
export type FormInput = { path: string; serializedPath: string; read?: unknown; profile: CsvProfile };
export type JsonSchema = {
  type?: string; enum?: unknown[]; default?: unknown; required?: string[];
  properties?: Record<string, JsonSchema>; items?: JsonSchema;
};
type FormSection = {
  path: string; widget: "column-map" | "utility-terms" | "nests" | "parameter-settings" | "object";
  input?: string; title?: string; labels?: Record<string, string>;
  allow_constant?: boolean; show_coefficient?: boolean; alternatives_from?: string;
  parameters_from?: string;
};
export type FormDefinition = {
  inputs?: Record<string, { title?: string; description?: string }>;
  sections?: FormSection[];
  defaults?: Record<string, unknown>;
};
type EditorDefinition = {
  config_schema: JsonSchema;
  ui_schema: { "ui:form"?: FormDefinition };
};
type DistinctValues = { values: string[]; rows_sampled: number; truncated: boolean };

export function SchemaFormEditor({ definition, inputs, config, onChooseInput, onConfigChange }: {
  definition: EditorDefinition;
  inputs: Record<string, FormInput | undefined>;
  config: Record<string, unknown>;
  onChooseInput: (slot: string) => void;
  onConfigChange: (config: Record<string, unknown>) => void;
}) {
  const form = definition.ui_schema["ui:form"];
  const sections = form?.sections ?? [];
  const [alternatives, setAlternatives] = useState<DistinctValues>();
  const alternativePath = sections.find((section) => section.alternatives_from)?.alternatives_from;
  const alternativeOwner = alternativePath?.split(".").slice(0, -1).join(".");
  const alternativeColumn = alternativePath ? getAt(config, alternativePath) : undefined;
  const alternativeSection = sections.find((section) => section.path === alternativeOwner);
  const alternativeInput = alternativeSection?.input ? inputs[alternativeSection.input] : undefined;

  useEffect(() => {
    let active = true;
    if (alternativeInput && typeof alternativeColumn === "string" && alternativeColumn) {
      invoke<DistinctValues>("inspect_distinct_values", {
        path: alternativeInput.path,
        column: alternativeColumn,
      }).then((value) => { if (active) setAlternatives(value); })
        .catch(() => { if (active) setAlternatives(undefined); });
    } else {
      setAlternatives(undefined);
    }
    return () => { active = false; };
  }, [alternativeInput?.path, alternativeColumn]);

  function update(path: string, value: unknown) {
    onConfigChange(setAt(config, path, value));
  }

  return <>
    <div className="subsection-heading"><div><h3>入力データ</h3><p>必要な入力slotはComponentManifestから読み込みます．</p></div></div>
    <div className="manifest-input-grid">{Object.entries(form?.inputs ?? {}).map(([slot, metadata]) => {
      const input = inputs[slot];
      return <article className="manifest-input" key={slot}>
        <div><strong>{metadata.title ?? slot}</strong><code>{slot}</code></div>
        {metadata.description && <p>{metadata.description}</p>}
        {input ? <><small>{input.path}</small><div className="profile-summary"><span>{input.profile.encoding}</span><span>{input.profile.rows_sampled}行確認</span><span>{input.profile.columns.length}列</span></div></> : <div className="empty-state compact-empty">未選択</div>}
        <button className="secondary" onClick={() => onChooseInput(slot)}>{input ? "変更" : "CSVを選択"}</button>
      </article>;
    })}</div>

    {sections.map((section) => {
      const schema = schemaAt(definition.config_schema, section.path);
      const value = getAt(config, section.path);
      const input = section.input ? inputs[section.input] : undefined;
      return <div className="manifest-section" key={section.path}>
        {section.widget === "column-map" && <ColumnMap section={section} schema={schema} value={asObject(value)} input={input} onChange={(next) => update(section.path, next)} />}
        {section.widget === "utility-terms" && <UtilityTerms section={section} value={asArray(value)} input={input} candidates={alternatives?.values ?? []} onChange={(next) => update(section.path, next)} />}
        {section.widget === "nests" && <NestEditor section={section} value={asArray(value)} candidates={alternatives?.values ?? []} onChange={(next) => update(section.path, next)} />}
        {section.widget === "parameter-settings" && <ParameterSettings section={section} value={asObject(value)} config={config} onChange={(next) => update(section.path, next)} />}
        {section.widget === "object" && <ObjectEditor section={section} schema={schema} value={asObject(value)} onChange={(next) => update(section.path, next)} />}
      </div>;
    })}
    {alternatives && <p className="hint">候補値: {alternatives.values.length}件，{alternatives.rows_sampled}行から取得{alternatives.truncated ? " (上限付きsample)" : ""}</p>}
  </>;
}

function SectionTitle({ section }: { section: FormSection }) {
  return <div className="subsection-heading"><div><h3>{section.title ?? section.path}</h3><p><code>{section.path}</code>としてAnalysis planへ保存します．</p></div></div>;
}

function ColumnMap({ section, schema, value, input, onChange }: {
  section: FormSection; schema?: JsonSchema; value: Record<string, unknown>; input?: FormInput;
  onChange: (value: Record<string, unknown>) => void;
}) {
  const names = Object.keys(schema?.properties ?? {});
  const required = schema?.required ?? [];
  return <><SectionTitle section={section} /><div className="role-grid">{names.map((name) => <label key={name}><span>{section.labels?.[name] ?? name}{required.includes(name) ? " *" : " (任意)"}</span>
    <select value={typeof value[name] === "string" ? value[name] as string : ""} disabled={!input} onChange={(event) => onChange(event.target.value ? { ...value, [name]: event.target.value } : without(value, name))}>
      <option value="">列を選択</option>{input?.profile.columns.map((column) => <option key={column.name} value={column.name}>{column.name} ({column.inferred_type})</option>)}
    </select></label>)}</div></>;
}

function UtilityTerms({ section, value, input, candidates, onChange }: {
  section: FormSection; value: unknown[]; input?: FormInput; candidates: string[];
  onChange: (value: unknown[]) => void;
}) {
  const terms = value.map(asObject);
  function patch(index: number, change: Record<string, unknown>) {
    onChange(terms.map((term, current) => current === index ? { ...term, ...change } : term));
  }
  return <><div className="subsection-heading"><div><h3>{section.title ?? section.path}</h3><p>parameterと効用へ入れる列を指定します．</p></div><button className="secondary" onClick={() => onChange([...terms, { parameter: `beta_${terms.length + 1}`, column: "", ...(section.show_coefficient ? { coefficient: 1 } : {}) }])}>termを追加</button></div>
    {!terms.length && <div className="empty-state">効用termを追加してください．</div>}
    {terms.map((term, index) => {
      const usesConstant = "constant" in term;
      return <div className={`term-row ${section.show_coefficient ? "with-coefficient" : ""}`} key={index}>
        <input aria-label="parameter" value={String(term.parameter ?? "")} onChange={(event) => patch(index, { parameter: event.target.value })} placeholder="beta_time" />
        {section.allow_constant && <select value={usesConstant ? "constant" : "column"} onChange={(event) => patch(index, event.target.value === "constant" ? { constant: 1, column: undefined } : { column: "", constant: undefined })}><option value="column">列</option><option value="constant">定数</option></select>}
        {usesConstant ? <input type="number" value={String(term.constant ?? 1)} onChange={(event) => patch(index, { constant: numberOrEmpty(event.target.value) })} />
          : <select value={String(term.column ?? "")} disabled={!input} onChange={(event) => patch(index, { column: event.target.value })}><option value="">列を選択</option>{input?.profile.columns.map((column) => <option key={column.name} value={column.name}>{column.name} ({column.inferred_type})</option>)}</select>}
        {section.show_coefficient && <input type="number" title="coefficient" value={String(term.coefficient ?? 1)} onChange={(event) => patch(index, { coefficient: numberOrEmpty(event.target.value) })} />}
        {section.alternatives_from && <AlternativePicker candidates={candidates} selected={(term.alternatives as unknown[] | undefined)?.map(String) ?? []} onChange={(selected) => patch(index, { alternatives: selected.length ? selected : undefined })} />}
        <button className="text-button danger-text" onClick={() => onChange(terms.filter((_, current) => current !== index))}>削除</button>
      </div>;
    })}</>;
}

function NestEditor({ section, value, candidates, onChange }: {
  section: FormSection; value: unknown[]; candidates: string[]; onChange: (value: unknown[]) => void;
}) {
  const nests = value.map(asObject);
  function patch(index: number, change: Record<string, unknown>) {
    onChange(nests.map((nest, current) => current === index ? { ...nest, ...change } : nest));
  }
  return <><div className="subsection-heading"><div><h3>{section.title ?? section.path}</h3><p>各選択肢を重複なく1つのnestへ入れます．</p></div><button className="secondary" onClick={() => onChange([...nests, { name: `nest_${nests.length + 1}`, alternatives: [], dissimilarity: { initial: 0.8 } }])}>Nestを追加</button></div>
    {!nests.length && <div className="empty-state">2つ以上のnestを追加してください．</div>}
    {nests.map((nest, index) => {
      const dissimilarity = asObject(nest.dissimilarity);
      const fixed = "fixed" in dissimilarity;
      return <div className="nest-row" key={index}><input value={String(nest.name ?? "")} onChange={(event) => patch(index, { name: event.target.value })} placeholder="motorized" />
        <AlternativePicker candidates={candidates} selected={(nest.alternatives as unknown[] | undefined)?.map(String) ?? []} onChange={(selected) => patch(index, { alternatives: selected })} />
        <select value={fixed ? "fixed" : "initial"} onChange={(event) => patch(index, { dissimilarity: event.target.value === "fixed" ? { fixed: 1 } : { initial: 0.8 } })}><option value="initial">推定</option><option value="fixed">固定</option></select>
        <input type="number" min="0.05" max="1" step="0.05" value={String(dissimilarity[fixed ? "fixed" : "initial"] ?? (fixed ? 1 : 0.8))} onChange={(event) => patch(index, { dissimilarity: { [fixed ? "fixed" : "initial"]: numberOrEmpty(event.target.value) } })} />
        <button className="text-button danger-text" onClick={() => onChange(nests.filter((_, current) => current !== index))}>削除</button></div>;
    })}</>;
}

function ParameterSettings({ section, value, config, onChange }: {
  section: FormSection; value: Record<string, unknown>; config: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}) {
  const terms = asArray(getAt(config, section.parameters_from ?? "terms")).map(asObject);
  const names = [...new Set(terms.map((term) => String(term.parameter ?? "")).filter(Boolean))];
  function update(name: string, field: string, raw: string) {
    const current = asObject(value[name]);
    const next = raw === "" ? without(current, field) : { ...current, [field]: Number(raw) };
    onChange({ ...value, [name]: next });
  }
  return <><SectionTitle section={section} />{!names.length && <div className="empty-state">先に効用termを追加してください．</div>}
    {names.map((name) => { const settings = asObject(value[name]); return <div className="parameter-row" key={name}><strong>{name}</strong>{["initial", "lower", "upper"].map((field) => <label key={field}><span>{field}</span><input type="number" value={settings[field] === undefined ? "" : String(settings[field])} placeholder="任意" onChange={(event) => update(name, field, event.target.value)} /></label>)}</div>; })}</>;
}

function ObjectEditor({ section, schema, value, onChange }: {
  section: FormSection; schema?: JsonSchema; value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}) {
  return <><SectionTitle section={section} /><div className="field-grid compact-grid">{Object.entries(schema?.properties ?? {}).map(([name, property]) => <label key={name}><span>{name}</span>{property.enum
    ? <select value={String(value[name] ?? property.default ?? "")} onChange={(event) => onChange({ ...value, [name]: event.target.value })}>{property.enum.map((item) => <option key={String(item)}>{String(item)}</option>)}</select>
    : <input type={property.type === "number" || property.type === "integer" ? "number" : "text"} value={String(value[name] ?? property.default ?? "")} onChange={(event) => onChange({ ...value, [name]: property.type === "number" || property.type === "integer" ? numberOrEmpty(event.target.value) : event.target.value })} />}</label>)}</div></>;
}

function AlternativePicker({ candidates, selected, onChange }: { candidates: string[]; selected: string[]; onChange: (value: string[]) => void }) {
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const visible = candidates.filter((candidate) => candidate.toLocaleLowerCase().includes(search.toLocaleLowerCase())).slice(0, search ? 50 : 8);
  function toggle(candidate: string) { onChange(selected.includes(candidate) ? selected.filter((item) => item !== candidate) : [...selected, candidate]); }
  return <div className="alternative-picker"><div className="alternative-control">{selected.map((item) => <button type="button" className="selected-tag" key={item} onMouseDown={(event) => event.preventDefault()} onClick={() => toggle(item)}>{item} ×</button>)}
    <input value={search} onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onChange={(event) => setSearch(event.target.value)} placeholder={selected.length ? "検索…" : candidates.length ? "値を検索" : "候補列を先に指定"} disabled={!candidates.length} /></div>
    {open && !!visible.length && <div className="alternative-menu">{visible.map((candidate) => <button type="button" className={selected.includes(candidate) ? "selected" : ""} key={candidate} onMouseDown={(event) => event.preventDefault()} onClick={() => toggle(candidate)}>{candidate}</button>)}</div>}</div>;
}

function getAt(value: unknown, path: string): unknown {
  return path.split(".").reduce<unknown>((current, key) => asObject(current)[key], value);
}
function setAt(root: Record<string, unknown>, path: string, value: unknown): Record<string, unknown> {
  const [head, ...tail] = path.split(".");
  return { ...root, [head]: tail.length ? setAt(asObject(root[head]), tail.join("."), value) : value };
}
function schemaAt(schema: JsonSchema, path: string): JsonSchema | undefined {
  return path.split(".").reduce<JsonSchema | undefined>((current, key) => current?.properties?.[key], schema);
}
function asObject(value: unknown): Record<string, unknown> { return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {}; }
function asArray(value: unknown): unknown[] { return Array.isArray(value) ? value : []; }
function without(value: Record<string, unknown>, key: string) { const next = { ...value }; delete next[key]; return next; }
function numberOrEmpty(value: string) { return value === "" ? undefined : Number(value); }
