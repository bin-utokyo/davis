export type AlternativeId = "car" | "rail" | "bus" | "walk";

export interface Dataset {
  id: string;
  name: string;
  observations: number;
  alternatives: number;
  description: string;
}

export interface UtilityTerm {
  id: string;
  coefficient: string;
  variable: string;
  kind: "constant" | "variable";
}

export interface AlternativeUtility {
  id: AlternativeId;
  label: string;
  color: string;
  terms: UtilityTerm[];
}

export interface ModelSpecification {
  alternatives: AlternativeUtility[];
}

export type SuggestionId = "share-cost" | "add-asc" | "income-car";

export interface Suggestion {
  id: SuggestionId;
  eyebrow: string;
  title: string;
  description: string;
  impact: "High" | "Medium" | "Explore";
}

export interface CoefficientResult {
  parameter: string;
  estimate: number;
  standardError: number;
  tValue: number;
  lower95: number;
  upper95: number;
}

export interface EstimationMetrics {
  logLikelihood: number;
  rhoSquared: number;
  aic: number;
  bic: number;
}

export interface EstimationResult {
  status: "converged";
  metrics: EstimationMetrics;
  coefficients: CoefficientResult[];
  iterations: number;
  gradientNorm: number;
  elapsedSeconds: number;
  diagnostics: string[];
}

export interface Experiment {
  id: string;
  name: string;
  summary: string;
  createdAt: string;
  specification: ModelSpecification;
  result: EstimationResult;
}

export type ResultTab = "table" | "coefficients" | "diagnostics";
export type ModelStatus = "Draft" | "Modified" | "Estimated" | "Saved";
