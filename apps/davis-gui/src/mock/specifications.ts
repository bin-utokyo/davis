import type {
  AlternativeId,
  AlternativeUtility,
  Dataset,
  ModelSpecification,
  UtilityTerm,
} from "../types";
import { datasets } from "./datasets";

const term = (
  id: string,
  coefficient: string,
  variable = "",
  kind: UtilityTerm["kind"] = "variable",
): UtilityTerm => ({ id, coefficient, variable, kind });

const columnFor = (
  dataset: Dataset,
  concept: "time" | "cost" | "income",
  alternativeId?: AlternativeId,
) => dataset.columns.find(
  (column) =>
    column.role === "explanatory" &&
    column.concept === concept &&
    (alternativeId === undefined || column.alternativeId === alternativeId),
);

export const createDefaultAlternative = (
  dataset: Dataset,
  id: AlternativeId,
): AlternativeUtility => {
  const metadata = dataset.alternatives.find((item) => item.id === id);
  if (!metadata) throw new Error(`Alternative ${id} is not available in ${dataset.id}`);

  const terms: UtilityTerm[] = [];
  if (id === "car") terms.push(term(`${id}-asc`, `ASC_${id}`, "", "constant"));

  const timeColumn = columnFor(dataset, "time", id);
  if (timeColumn) terms.push(term(`${id}-time`, "beta_time", timeColumn.name));

  const costColumn = columnFor(dataset, "cost", id);
  if (id === "car" && costColumn) {
    terms.push(term(`${id}-cost`, "beta_cost", costColumn.name));
  }

  return { ...metadata, terms };
};

export const createInitialSpecification = (dataset: Dataset): ModelSpecification => ({
  alternatives: dataset.alternatives.map((item) => createDefaultAlternative(dataset, item.id)),
});

const tokyoDataset = datasets[0];

const createTimeOnlySpecification = (dataset: Dataset): ModelSpecification => ({
  alternatives: dataset.alternatives.map((item) => {
    const timeColumn = columnFor(dataset, "time", item.id);
    return {
      ...item,
      terms: timeColumn ? [term(`${item.id}-time`, "beta_time", timeColumn.name)] : [],
    };
  }),
});

export const timeOnlySpecification = createTimeOnlySpecification(tokyoDataset);

export const initialSpecification = createInitialSpecification(tokyoDataset);

export const timeCostSpecification: ModelSpecification = {
  alternatives: createTimeOnlySpecification(tokyoDataset).alternatives.map((item) => {
    const costColumn = columnFor(tokyoDataset, "cost", item.id);
    return {
      ...item,
      terms: costColumn
        ? [...item.terms, term(`${item.id}-cost`, "beta_cost", costColumn.name)]
        : item.terms,
    };
  }),
};

export const fullSpecification: ModelSpecification = {
  alternatives: timeCostSpecification.alternatives.map((item, index, all) => ({
    ...item,
    terms: index < all.length - 1
      ? [term(`${item.id}-asc`, `ASC_${item.id}`, "", "constant"), ...item.terms]
      : item.terms,
  })),
};

export const cloneSpecification = (specification: ModelSpecification): ModelSpecification =>
  structuredClone(specification);
