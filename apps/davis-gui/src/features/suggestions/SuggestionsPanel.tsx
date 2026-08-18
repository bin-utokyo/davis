import { ArrowRight, Check, Lightbulb, Sparkles } from "lucide-react";
import type { Suggestion, SuggestionId } from "../../types";

interface SuggestionsPanelProps {
  suggestions: Suggestion[];
  onApply: (suggestionId: SuggestionId) => void;
}

export function SuggestionsPanel({ suggestions, onApply }: SuggestionsPanelProps) {
  return (
    <aside className="panel suggestions-panel" aria-labelledby="suggestions-heading">
      <div className="panel-heading suggestions-heading-row">
        <div>
          <span className="section-kicker">Model review</span>
          <h2 id="suggestions-heading">
            <Sparkles aria-hidden="true" size={17} />Suggestions
            <span className="count-badge">{suggestions.length}</span>
          </h2>
        </div>
      </div>
      <div className="suggestion-list">
        {suggestions.length ? (
          suggestions.map((suggestion, index) => (
            <article className="suggestion-card" key={suggestion.id}>
              <div className="suggestion-index"><Lightbulb aria-hidden="true" size={16} /></div>
              <div className="suggestion-content">
                <div className="suggestion-meta">
                  <span>{suggestion.eyebrow}</span>
                  <em data-impact={suggestion.impact.toLowerCase()}>{suggestion.impact}</em>
                </div>
                <h3>{suggestion.title}</h3>
                <p>{suggestion.description}</p>
                <button
                  className="apply-button"
                  onClick={() => onApply(suggestion.id)}
                  aria-label={`Apply suggestion: ${suggestion.title}`}
                >
                  Apply <ArrowRight aria-hidden="true" size={14} />
                </button>
              </div>
              <span className="suggestion-number">0{index + 1}</span>
            </article>
          ))
        ) : (
          <div className="suggestions-empty">
            <span><Check aria-hidden="true" size={19} /></span>
            <strong>No open suggestions</strong>
            <p>This specification covers the current mock review rules.</p>
          </div>
        )}
      </div>
      <div className="suggestions-footer">
        Suggestions are local heuristics for UX testing, not statistical advice.
      </div>
    </aside>
  );
}
