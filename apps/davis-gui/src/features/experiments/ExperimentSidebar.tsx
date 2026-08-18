import { FlaskConical, GitCompareArrows, Plus, TimerReset } from "lucide-react";
import type { Experiment } from "../../types";

interface ExperimentSidebarProps {
  experiments: Experiment[];
  activeExperimentId?: string;
  onSelect: (experiment: Experiment) => void;
  onNew: () => void;
  onCompare: () => void;
}

export function ExperimentSidebar({
  experiments,
  activeExperimentId,
  onSelect,
  onNew,
  onCompare,
}: ExperimentSidebarProps) {
  return (
    <aside className="panel experiments-panel" aria-label="Experiment history">
      <div className="panel-heading experiments-heading">
        <div>
          <span className="section-kicker">Workspace</span>
          <h2>Experiments</h2>
        </div>
        <button className="icon-button" aria-label="Create new MNL model" onClick={onNew}>
          <Plus aria-hidden="true" size={17} />
        </button>
      </div>
      <div className="experiment-list" id="experiment-list">
        {experiments.map((experiment) => (
          <button
            key={experiment.id}
            className="experiment-card"
            data-active={activeExperimentId === experiment.id}
            onClick={() => onSelect(experiment)}
          >
            <span className="experiment-marker"><FlaskConical aria-hidden="true" size={15} /></span>
            <span className="experiment-copy">
              <span className="experiment-title-row">
                <strong>{experiment.name}</strong>
                {activeExperimentId === experiment.id && <i>Active</i>}
              </span>
              <span>{experiment.summary}</span>
              <small><TimerReset aria-hidden="true" size={12} />{experiment.createdAt}</small>
            </span>
          </button>
        ))}
      </div>
      <div className="experiment-actions">
        <button className="secondary-button full-button" onClick={onCompare}>
          <GitCompareArrows aria-hidden="true" size={15} />
          Compare models
        </button>
      </div>
    </aside>
  );
}
