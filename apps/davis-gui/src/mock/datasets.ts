import type { Dataset } from "../types";

export const datasets: Dataset[] = [
  {
    id: "tokyo-pt-2018",
    name: "Tokyo PT 2018",
    observations: 43_812,
    alternatives: 4,
    description: "Metropolitan person-trip survey",
  },
  {
    id: "synthetic-commute",
    name: "Synthetic Commute",
    observations: 12_400,
    alternatives: 4,
    description: "Calibrated commuting choices",
  },
  {
    id: "teaching-sample",
    name: "Teaching Sample",
    observations: 2_500,
    alternatives: 4,
    description: "Compact classroom dataset",
  },
];
