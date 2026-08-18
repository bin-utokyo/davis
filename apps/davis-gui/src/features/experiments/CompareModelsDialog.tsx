import * as Dialog from "@radix-ui/react-dialog";
import { CheckCircle2, X } from "lucide-react";
import type { Experiment } from "../../types";

interface CompareModelsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  experiments: Experiment[];
}

export function CompareModelsDialog({
  open,
  onOpenChange,
  experiments,
}: CompareModelsDialogProps) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="compare-dialog">
          <div className="compare-title-row">
            <div>
              <span className="section-kicker">Experiment review</span>
              <Dialog.Title>Compare models</Dialog.Title>
              <Dialog.Description>
                Follow fit improvements as the specification becomes richer.
              </Dialog.Description>
            </div>
            <Dialog.Close asChild>
              <button className="icon-button" aria-label="Close model comparison">
                <X aria-hidden="true" size={17} />
              </button>
            </Dialog.Close>
          </div>
          <div className="compare-grid">
            {experiments.map((experiment, index) => (
              <article key={experiment.id} className="compare-card" data-best={index === 0}>
                <div className="compare-card-title">
                  <span>
                    <strong>{experiment.name}</strong>
                    <small>{experiment.summary}</small>
                  </span>
                  {index === 0 && <span className="best-badge"><CheckCircle2 size={13} />Best fit</span>}
                </div>
                <dl>
                  <div><dt>LL</dt><dd>{experiment.result.metrics.logLikelihood.toFixed(1)}</dd></div>
                  <div><dt>ρ²</dt><dd>{experiment.result.metrics.rhoSquared.toFixed(2)}</dd></div>
                  <div><dt>AIC</dt><dd>{experiment.result.metrics.aic}</dd></div>
                  <div><dt>BIC</dt><dd>{experiment.result.metrics.bic}</dd></div>
                </dl>
              </article>
            ))}
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
