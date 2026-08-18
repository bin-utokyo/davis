import {
  CheckCircle2,
  Command,
  LoaderCircle,
  Moon,
  Play,
  Sun,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { CommandPalette, type CommandId } from "./features/command-palette/CommandPalette";
import { DatasetSelector } from "./features/datasets/DatasetSelector";
import { CompareModelsDialog } from "./features/experiments/CompareModelsDialog";
import { ExperimentSidebar } from "./features/experiments/ExperimentSidebar";
import { UtilityEditor } from "./features/model-editor/UtilityEditor";
import { ResultsPanel } from "./features/results/ResultsPanel";
import { SuggestionsPanel } from "./features/suggestions/SuggestionsPanel";
import {
  cloneSpecification,
  createDefaultAlternative,
  createInitialSpecification,
  initialSpecification,
} from "./mock/specifications";
import { getDatasets } from "./services/datasetService";
import { runEstimation } from "./services/estimationService";
import { getExperiments } from "./services/experimentService";
import { applySuggestion, getSuggestions } from "./services/suggestionService";
import type {
  AlternativeId,
  Dataset,
  EstimationResult,
  Experiment,
  ModelSpecification,
  ModelStatus,
  ResultTab,
  Suggestion,
  SuggestionId,
} from "./types";

const scrollToResults = () => {
  window.requestAnimationFrame(() => {
    document.getElementById("results-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
};

type Theme = "light" | "dark";

const getInitialTheme = (): Theme => {
  const savedTheme = window.localStorage.getItem("davis-theme");
  return savedTheme === "light" || savedTheme === "dark" ? savedTheme : "dark";
};

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<Dataset>();
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [activeExperimentId, setActiveExperimentId] = useState<string>();
  const [specification, setSpecification] = useState<ModelSpecification>(() =>
    cloneSpecification(initialSpecification),
  );
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [result, setResult] = useState<EstimationResult>();
  const [resultTab, setResultTab] = useState<ResultTab>("coefficients");
  const [modelStatus, setModelStatus] = useState<ModelStatus>("Draft");
  const [estimationState, setEstimationState] = useState<"idle" | "running" | "converged">("idle");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [announcement, setAnnouncement] = useState("DAVIS GUI loaded");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("davis-theme", theme);
    document
      .querySelector('meta[name="theme-color"]')
      ?.setAttribute("content", theme === "dark" ? "#0d0c12" : "#f6f5fa");
  }, [theme]);

  useEffect(() => {
    let active = true;
    Promise.all([getDatasets(), getExperiments()]).then(([nextDatasets, nextExperiments]) => {
      if (!active) return;
      setDatasets(nextDatasets);
      setExperiments(nextExperiments);
      if (nextExperiments[0]) {
        setSelectedDataset(
          nextDatasets.find((dataset) => dataset.id === nextExperiments[0].datasetId) ?? nextDatasets[0],
        );
        setActiveExperimentId(nextExperiments[0].id);
        setSpecification(cloneSpecification(nextExperiments[0].specification));
        setResult(nextExperiments[0].result);
        setEstimationState("converged");
        setModelStatus("Saved");
      } else {
        setSelectedDataset(nextDatasets[0]);
        if (nextDatasets[0]) {
          setSpecification(createInitialSpecification(nextDatasets[0]));
        }
      }
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
    let active = true;
    getSuggestions(specification, selectedDataset).then((nextSuggestions) => {
      if (active) setSuggestions(nextSuggestions);
    });
    return () => {
      active = false;
    };
  }, [selectedDataset, specification]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const currentModelName = useMemo(
    () => experiments.find((experiment) => experiment.id === activeExperimentId)?.name ?? "Untitled MNL",
    [activeExperimentId, experiments],
  );

  const markModified = useCallback((nextSpecification: ModelSpecification) => {
    setSpecification(nextSpecification);
    setActiveExperimentId(undefined);
    setResult(undefined);
    setEstimationState("idle");
    setModelStatus("Modified");
  }, []);

  const createNewModel = useCallback(() => {
    setSpecification(
      selectedDataset
        ? createInitialSpecification(selectedDataset)
        : cloneSpecification(initialSpecification),
    );
    setActiveExperimentId(undefined);
    setResult(undefined);
    setResultTab("table");
    setEstimationState("idle");
    setModelStatus("Draft");
    setAnnouncement("New MNL model created");
  }, [selectedDataset]);

  const openExperiment = useCallback((experiment: Experiment) => {
    const experimentDataset = datasets.find((dataset) => dataset.id === experiment.datasetId);
    if (experimentDataset) setSelectedDataset(experimentDataset);
    setSpecification(cloneSpecification(experiment.specification));
    setActiveExperimentId(experiment.id);
    setResult(structuredClone(experiment.result));
    setResultTab("coefficients");
    setEstimationState("converged");
    setModelStatus("Saved");
    setAnnouncement(`${experiment.name} restored`);
  }, [datasets]);

  const selectDataset = useCallback((dataset: Dataset) => {
    setSelectedDataset(dataset);
    setSpecification(createInitialSpecification(dataset));
    setActiveExperimentId(undefined);
    setResult(undefined);
    setResultTab("table");
    setEstimationState("idle");
    setModelStatus("Draft");
    setAnnouncement(`${dataset.name} selected; specification reset to its schema`);
  }, []);

  const handleCommand = useCallback((command: CommandId) => {
    if (command === "new-model") createNewModel();
    if (command === "choose-dataset") setDatasetOpen(true);
    if (command === "compare-models") setCompareOpen(true);
    if (command === "open-experiment") {
      window.requestAnimationFrame(() => {
        const firstExperiment = document.querySelector<HTMLButtonElement>(".experiment-card");
        firstExperiment?.focus();
        firstExperiment?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
    if (command === "estimation-table") {
      setResultTab("table");
      scrollToResults();
    }
    if (command === "plot-coefficients") {
      setResultTab("coefficients");
      scrollToResults();
    }
  }, [createNewModel]);

  const updateTerm = (
    alternativeId: AlternativeId,
    termId: string,
    field: "coefficient" | "variable",
    value: string,
  ) => {
    if (
      field === "variable" &&
      !selectedDataset?.columns.some(
        (column) => column.role === "explanatory" && column.name === value,
      )
    ) return;
    const next = cloneSpecification(specification);
    const term = next.alternatives
      .find((alternative) => alternative.id === alternativeId)
      ?.terms.find((item) => item.id === termId);
    if (!term) return;
    term[field] = value;
    markModified(next);
  };

  const addTerm = (alternativeId: AlternativeId, variable: string) => {
    if (!selectedDataset?.columns.some(
      (column) => column.role === "explanatory" && column.name === variable,
    )) return;
    const next = cloneSpecification(specification);
    const alternative = next.alternatives.find((item) => item.id === alternativeId);
    if (!alternative) return;
    if (alternative.terms.some((term) => term.variable === variable)) return;
    const coefficientSuffix = variable.replace(/[^a-zA-Z0-9_]/g, "_").toLowerCase();
    alternative.terms.push({
      id: `${alternativeId}-custom-${Date.now()}`,
      coefficient: `beta_${coefficientSuffix}`,
      variable,
      kind: "variable",
    });
    markModified(next);
  };

  const toggleAlternative = (alternativeId: AlternativeId) => {
    if (!selectedDataset) return;
    const isActive = specification.alternatives.some((item) => item.id === alternativeId);
    if (isActive && specification.alternatives.length <= 2) return;

    if (isActive) {
      const nextAlternatives = cloneSpecification(specification).alternatives.filter(
        (item) => item.id !== alternativeId,
      );
      if (specification.alternatives.at(-1)?.id === alternativeId) {
        const nextReference = nextAlternatives.at(-1);
        if (nextReference) {
          nextReference.terms = nextReference.terms.filter(
            (term) => term.coefficient !== `ASC_${nextReference.id}`,
          );
        }
      }
      markModified({
        alternatives: nextAlternatives,
      });
      const label = selectedDataset.alternatives.find((item) => item.id === alternativeId)?.label;
      setAnnouncement(`${label ?? alternativeId} removed from the choice set`);
      return;
    }

    const existingById = new Map(
      specification.alternatives.map((alternative) => [alternative.id, alternative]),
    );
    existingById.set(
      alternativeId,
      createDefaultAlternative(selectedDataset, alternativeId),
    );
    markModified({
      alternatives: selectedDataset.alternatives
        .map((item) => existingById.get(item.id))
        .filter((item): item is NonNullable<typeof item> => Boolean(item)),
    });
    const label = selectedDataset.alternatives.find((item) => item.id === alternativeId)?.label;
    setAnnouncement(`${label ?? alternativeId} added to the choice set`);
  };

  const removeTerm = (alternativeId: AlternativeId, termId: string) => {
    const next = cloneSpecification(specification);
    const alternative = next.alternatives.find((item) => item.id === alternativeId);
    if (!alternative) return;
    alternative.terms = alternative.terms.filter((item) => item.id !== termId);
    markModified(next);
  };

  const handleSuggestion = (suggestionId: SuggestionId) => {
    if (!selectedDataset) return;
    markModified(applySuggestion(specification, suggestionId, selectedDataset));
    const suggestion = suggestions.find((item) => item.id === suggestionId);
    setAnnouncement(`${suggestion?.title ?? "Suggestion"} applied`);
  };

  const handleRunEstimation = async () => {
    if (estimationState === "running") return;
    setEstimationState("running");
    setAnnouncement("Mock estimation running");
    const nextResult = await runEstimation(specification);
    setResult(nextResult);
    setEstimationState("converged");
    setModelStatus("Estimated");
    setResultTab("table");
    setAnnouncement("Mock estimation converged");
    scrollToResults();
  };

  const handleThemeToggle = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    setAnnouncement(`${nextTheme === "light" ? "Light" : "Dark"} theme enabled`);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block" aria-label="DAVIS GUI">
          <span className="brand-mark">
            <img src="/bin-mark.svg" alt="" />
          </span>
          <span><strong>DAVIS</strong><small>BIN · Choice Lab</small></span>
        </div>
        <DatasetSelector
          datasets={datasets}
          selected={selectedDataset}
          open={datasetOpen}
          onOpenChange={setDatasetOpen}
          onSelect={selectDataset}
        />
        <button className="command-trigger" onClick={() => setPaletteOpen(true)}>
          <Command aria-hidden="true" size={15} />
          <span>Search or run a command</span>
          <kbd>{navigator.platform.includes("Mac") ? "⌘" : "Ctrl"} K</kbd>
        </button>
        <div className="model-context">
          <span>{currentModelName}</span>
          <small>{modelStatus}</small>
        </div>
        <div className="run-area">
          <button
            className="theme-toggle"
            type="button"
            onClick={handleThemeToggle}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            {theme === "dark" ? (
              <Sun aria-hidden="true" size={17} />
            ) : (
              <Moon aria-hidden="true" size={17} />
            )}
          </button>
          {estimationState === "converged" && (
            <span className="top-status"><CheckCircle2 aria-hidden="true" size={14} />Converged</span>
          )}
          <button
            className="run-button"
            onClick={handleRunEstimation}
            disabled={estimationState === "running"}
          >
            {estimationState === "running" ? (
              <><LoaderCircle className="spin" aria-hidden="true" size={16} />Estimating...</>
            ) : (
              <><Play aria-hidden="true" size={15} />Run Estimation</>
            )}
          </button>
        </div>
      </header>

      <main className="workspace">
        <ExperimentSidebar
          experiments={experiments}
          activeExperimentId={activeExperimentId}
          onSelect={openExperiment}
          onNew={createNewModel}
          onCompare={() => setCompareOpen(true)}
        />
        <UtilityEditor
          dataset={selectedDataset}
          specification={specification}
          status={modelStatus}
          onTermChange={updateTerm}
          onAddTerm={addTerm}
          onRemoveTerm={removeTerm}
          onAlternativeToggle={toggleAlternative}
        />
        <SuggestionsPanel suggestions={suggestions} onApply={handleSuggestion} />
        <ResultsPanel result={result} activeTab={resultTab} onTabChange={setResultTab} />
      </main>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} onCommand={handleCommand} />
      <CompareModelsDialog open={compareOpen} onOpenChange={setCompareOpen} experiments={experiments} />
      <div className="sr-only" aria-live="polite">{announcement}</div>
    </div>
  );
}
