import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import {
  Braces,
  Check,
  ChevronDown,
  Columns3,
  Plus,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import type {
  AlternativeId,
  Dataset,
  DatasetColumn,
  ModelSpecification,
  ModelStatus,
  UtilityTerm,
} from "../../types";

interface UtilityEditorProps {
  dataset?: Dataset;
  specification: ModelSpecification;
  status: ModelStatus;
  onTermChange: (
    alternativeId: AlternativeId,
    termId: string,
    field: "coefficient" | "variable",
    value: string,
  ) => void;
  onAddTerm: (alternativeId: AlternativeId, variable: string) => void;
  onRemoveTerm: (alternativeId: AlternativeId, termId: string) => void;
  onAlternativeToggle: (alternativeId: AlternativeId) => void;
}

function TermRow({
  alternativeId,
  columns,
  term,
  onTermChange,
  onRemoveTerm,
}: {
  alternativeId: AlternativeId;
  columns: DatasetColumn[];
  term: UtilityTerm;
  onTermChange: UtilityEditorProps["onTermChange"];
  onRemoveTerm: UtilityEditorProps["onRemoveTerm"];
}) {
  return (
    <div className="term-row">
      <span className="term-handle" aria-hidden="true">⋮⋮</span>
      <label>
        <span className="sr-only">Coefficient</span>
        <input
          value={term.coefficient}
          onChange={(event) =>
            onTermChange(alternativeId, term.id, "coefficient", event.target.value)
          }
          spellCheck={false}
          aria-label={`${alternativeId} coefficient`}
        />
      </label>
      {term.kind === "constant" ? (
        <span className="constant-pill">constant</span>
      ) : (
        <>
          <span className="multiply" aria-hidden="true">×</span>
          <label>
            <span className="sr-only">Dataset column</span>
            <select
              value={term.variable}
              onChange={(event) =>
                onTermChange(alternativeId, term.id, "variable", event.target.value)
              }
              aria-label={`${alternativeId} variable ${term.coefficient}`}
            >
              {columns.map((column) => (
                <option key={column.name} value={column.name}>
                  {column.name}
                </option>
              ))}
            </select>
          </label>
        </>
      )}
      <button
        className="term-delete"
        aria-label={`Remove ${term.coefficient} from ${alternativeId}`}
        onClick={() => onRemoveTerm(alternativeId, term.id)}
      >
        <Trash2 aria-hidden="true" size={13} />
      </button>
    </div>
  );
}

export function UtilityEditor({
  dataset,
  specification,
  status,
  onTermChange,
  onAddTerm,
  onRemoveTerm,
  onAlternativeToggle,
}: UtilityEditorProps) {
  const explanatoryColumns = dataset?.columns.filter(
    (column) => column.role === "explanatory" && column.type === "number",
  ) ?? [];
  const activeIds = new Set(specification.alternatives.map((alternative) => alternative.id));
  const referenceAlternative = specification.alternatives.at(-1);

  return (
    <section className="panel utility-panel" aria-labelledby="utility-heading">
      <div className="panel-heading utility-heading-row">
        <div>
          <span className="section-kicker">Multinomial logit</span>
          <h2 id="utility-heading"><Braces aria-hidden="true" size={18} />MNL Utility</h2>
        </div>
        <div className="utility-heading-actions">
          <span className="schema-badge" title="Explanatory variables are restricted to numeric dataset columns">
            <ShieldCheck aria-hidden="true" size={12} />
            Schema locked · {explanatoryColumns.length} columns
          </span>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <button className="alternative-trigger" aria-label="Select alternatives">
                <Columns3 aria-hidden="true" size={13} />
                Alternatives {specification.alternatives.length} / {dataset?.alternatives.length ?? 0}
                <ChevronDown aria-hidden="true" size={12} />
              </button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Portal>
              <DropdownMenu.Content
                className="dropdown-content alternatives-menu"
                sideOffset={7}
                align="end"
              >
                <div className="dropdown-label">Choice alternatives</div>
                {dataset?.alternatives.map((alternative) => {
                  const checked = activeIds.has(alternative.id);
                  const disabled = checked && specification.alternatives.length <= 2;
                  return (
                    <DropdownMenu.CheckboxItem
                      key={alternative.id}
                      className="alternative-option"
                      checked={checked}
                      disabled={disabled}
                      onCheckedChange={() => onAlternativeToggle(alternative.id)}
                      onSelect={(event) => event.preventDefault()}
                      data-testid={`alternative-option-${alternative.id}`}
                    >
                      <span className="alternative-check">
                        <DropdownMenu.ItemIndicator>
                          <Check aria-hidden="true" size={13} />
                        </DropdownMenu.ItemIndicator>
                      </span>
                      <span className="alternative-dot" style={{ backgroundColor: alternative.color }} />
                      <strong>{alternative.label}</strong>
                      {disabled && <small>Keep 2+</small>}
                    </DropdownMenu.CheckboxItem>
                  );
                })}
                <div className="alternatives-hint">An MNL model needs at least two alternatives.</div>
              </DropdownMenu.Content>
            </DropdownMenu.Portal>
          </DropdownMenu.Root>
          <span className="model-status" data-status={status.toLowerCase()}>{status}</span>
        </div>
      </div>
      <div className="utility-grid" data-alternative-count={specification.alternatives.length}>
        {specification.alternatives.map((alternative) => {
          const usedVariables = new Set(
            alternative.terms
              .filter((term) => term.kind === "variable")
              .map((term) => term.variable),
          );
          const availableColumns = explanatoryColumns.filter(
            (column) => !usedVariables.has(column.name),
          );

          return (
            <article
              key={alternative.id}
              className="alternative-card"
              data-testid={`utility-${alternative.id}`}
            >
              <header>
                <span className="alternative-dot" style={{ backgroundColor: alternative.color }} />
                <strong>{alternative.label}</strong>
                <small>U<sub>{alternative.label.toLowerCase()}</sub></small>
              </header>
              <div className="term-list">
                {alternative.terms.length === 0 && (
                  <span className="empty-utility">No utility terms selected</span>
                )}
                {alternative.terms.map((term) => (
                  <TermRow
                    key={term.id}
                    alternativeId={alternative.id}
                    columns={explanatoryColumns}
                    term={term}
                    onTermChange={onTermChange}
                    onRemoveTerm={onRemoveTerm}
                  />
                ))}
              </div>
              <label className="add-term-select">
                <Plus aria-hidden="true" size={14} />
                <span className="sr-only">Add explanatory variable to {alternative.label}</span>
                <select
                  value=""
                  onChange={(event) => {
                    if (event.target.value) onAddTerm(alternative.id, event.target.value);
                  }}
                  aria-label={`Add explanatory variable to ${alternative.label}`}
                  disabled={availableColumns.length === 0}
                >
                  <option value="">
                    {availableColumns.length > 0 ? "Add data column…" : "All columns added"}
                  </option>
                  {availableColumns.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name} — {column.label}
                    </option>
                  ))}
                </select>
              </label>
            </article>
          );
        })}
      </div>
      <footer className="utility-footer">
        <span>{referenceAlternative?.label ?? "No alternative"} is the reference alternative</span>
        <span>Variables are exact columns from {dataset?.name ?? "the dataset"}</span>
      </footer>
    </section>
  );
}
