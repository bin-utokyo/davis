import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  FileSpreadsheet,
  Gauge,
} from "lucide-react";
import type { EstimationResult, ResultTab } from "../../types";

interface ResultsPanelProps {
  result?: EstimationResult;
  activeTab: ResultTab;
  onTabChange: (tab: ResultTab) => void;
}

const formatEstimate = (value: number, digits = 3) => value.toFixed(digits);

const scalePosition = (value: number) => {
  const clamped = Math.max(-1.25, Math.min(1.25, value));
  return ((clamped + 1.25) / 2.5) * 100;
};

function EmptyResults() {
  return (
    <div className="results-empty">
      <span><Activity aria-hidden="true" size={20} /></span>
      <div>
        <strong>Ready to estimate</strong>
        <p>Run the mock estimator to populate model fit and coefficient results.</p>
      </div>
    </div>
  );
}

export function ResultsPanel({ result, activeTab, onTabChange }: ResultsPanelProps) {
  return (
    <section className="panel results-panel" id="results-panel" aria-labelledby="results-heading">
      <div className="results-topline">
        <div>
          <span className="section-kicker">Mock output</span>
          <h2 id="results-heading">Results</h2>
        </div>
        {result ? (
          <span className="converged-badge" data-testid="result-status">
            <CheckCircle2 aria-hidden="true" size={14} /> Converged
          </span>
        ) : (
          <span className="awaiting-badge">Not estimated</span>
        )}
      </div>
      <Tabs.Root
        className="results-tabs"
        value={activeTab}
        onValueChange={(value) => onTabChange(value as ResultTab)}
      >
        <Tabs.List className="tabs-list" aria-label="Result views">
          <Tabs.Trigger value="table"><FileSpreadsheet size={14} />Table</Tabs.Trigger>
          <Tabs.Trigger value="coefficients"><BarChart3 size={14} />Coefficients</Tabs.Trigger>
          <Tabs.Trigger value="diagnostics"><Gauge size={14} />Diagnostics</Tabs.Trigger>
        </Tabs.List>
        <div className="results-body">
          {!result ? (
            <EmptyResults />
          ) : (
            <>
              <Tabs.Content className="tab-content" value="table">
                <div className="metrics-strip">
                  <div><span>LL</span><strong>{result.metrics.logLikelihood.toFixed(1)}</strong></div>
                  <div><span>ρ²</span><strong>{result.metrics.rhoSquared.toFixed(2)}</strong></div>
                  <div><span>AIC</span><strong>{result.metrics.aic}</strong></div>
                  <div><span>BIC</span><strong>{result.metrics.bic}</strong></div>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Estimate</th>
                        <th>Std. error</th>
                        <th>t-value</th>
                        <th>95% CI</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.coefficients.map((coefficient) => (
                        <tr key={coefficient.parameter}>
                          <td><code>{coefficient.parameter}</code></td>
                          <td>{formatEstimate(coefficient.estimate, 4)}</td>
                          <td>{formatEstimate(coefficient.standardError, 4)}</td>
                          <td>{formatEstimate(coefficient.tValue, 2)}</td>
                          <td>
                            [{formatEstimate(coefficient.lower95, 3)}, {formatEstimate(coefficient.upper95, 3)}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Tabs.Content>
              <Tabs.Content className="tab-content" value="coefficients">
                <div className="coefficient-layout">
                  <div className="fit-card">
                    <span>Model fit</span>
                    <strong>ρ² {result.metrics.rhoSquared.toFixed(2)}</strong>
                    <p>Improvement over constants-only model</p>
                    <div className="fit-meter"><i style={{ width: `${result.metrics.rhoSquared * 100}%` }} /></div>
                  </div>
                  <div className="forest-plot" aria-label="Coefficient confidence intervals">
                    <div className="forest-axis"><span>-1.25</span><span>0</span><span>1.25</span></div>
                    {result.coefficients.map((coefficient) => {
                      const left = scalePosition(coefficient.lower95);
                      const right = scalePosition(coefficient.upper95);
                      const point = scalePosition(coefficient.estimate);
                      return (
                        <div className="forest-row" key={coefficient.parameter}>
                          <code>{coefficient.parameter}</code>
                          <div className="forest-track">
                            <i className="zero-line" />
                            <i
                              className="confidence-line"
                              style={{ left: `${left}%`, width: `${Math.max(1, right - left)}%` }}
                            />
                            <i className="estimate-dot" style={{ left: `${point}%` }} />
                          </div>
                          <span>{formatEstimate(coefficient.estimate, 3)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </Tabs.Content>
              <Tabs.Content className="tab-content" value="diagnostics">
                <div className="diagnostics-grid">
                  <dl className="diagnostic-stats">
                    <div><dt>Optimizer</dt><dd>BFGS</dd></div>
                    <div><dt>Iterations</dt><dd>{result.iterations}</dd></div>
                    <div><dt>Gradient norm</dt><dd>{result.gradientNorm.toExponential(1)}</dd></div>
                    <div><dt>Elapsed</dt><dd>{result.elapsedSeconds.toFixed(2)} s</dd></div>
                  </dl>
                  <div className="diagnostic-messages">
                    {result.diagnostics.map((message) => (
                      <div key={message}><CheckCircle2 aria-hidden="true" size={15} /><span>{message}</span></div>
                    ))}
                  </div>
                </div>
              </Tabs.Content>
            </>
          )}
        </div>
      </Tabs.Root>
    </section>
  );
}
