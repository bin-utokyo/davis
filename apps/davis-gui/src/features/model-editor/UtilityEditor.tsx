import { Braces, Plus, Trash2 } from "lucide-react";
import type {
  AlternativeId,
  ModelSpecification,
  ModelStatus,
  UtilityTerm,
} from "../../types";

interface UtilityEditorProps {
  specification: ModelSpecification;
  status: ModelStatus;
  onTermChange: (
    alternativeId: AlternativeId,
    termId: string,
    field: "coefficient" | "variable",
    value: string,
  ) => void;
  onAddTerm: (alternativeId: AlternativeId) => void;
  onRemoveTerm: (alternativeId: AlternativeId, termId: string) => void;
}

function TermRow({
  alternativeId,
  term,
  onTermChange,
  onRemoveTerm,
}: {
  alternativeId: AlternativeId;
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
            <span className="sr-only">Variable</span>
            <input
              value={term.variable}
              onChange={(event) =>
                onTermChange(alternativeId, term.id, "variable", event.target.value)
              }
              spellCheck={false}
              aria-label={`${alternativeId} variable`}
            />
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
  specification,
  status,
  onTermChange,
  onAddTerm,
  onRemoveTerm,
}: UtilityEditorProps) {
  return (
    <section className="panel utility-panel" aria-labelledby="utility-heading">
      <div className="panel-heading utility-heading-row">
        <div>
          <span className="section-kicker">Multinomial logit</span>
          <h2 id="utility-heading"><Braces aria-hidden="true" size={18} />MNL Utility</h2>
        </div>
        <span className="model-status" data-status={status.toLowerCase()}>{status}</span>
      </div>
      <div className="utility-grid">
        {specification.alternatives.map((alternative) => (
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
              {alternative.terms.map((term) => (
                <TermRow
                  key={term.id}
                  alternativeId={alternative.id}
                  term={term}
                  onTermChange={onTermChange}
                  onRemoveTerm={onRemoveTerm}
                />
              ))}
            </div>
            <button className="add-term-button" onClick={() => onAddTerm(alternative.id)}>
              <Plus aria-hidden="true" size={14} /> Add term
            </button>
          </article>
        ))}
      </div>
      <footer className="utility-footer">
        <span>Walk is the reference alternative</span>
        <span>Generic parameters are shared by name</span>
      </footer>
    </section>
  );
}
