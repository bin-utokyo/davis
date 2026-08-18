import { suggestionCatalog } from "../mock/suggestions";
import type {
  AlternativeId,
  Dataset,
  ModelSpecification,
  Suggestion,
  SuggestionId,
  UtilityTerm,
} from "../types";

const hasTerm = (
  specification: ModelSpecification,
  alternativeId: AlternativeId,
  coefficient: string,
) =>
  specification.alternatives
    .find((alternative) => alternative.id === alternativeId)
    ?.terms.some((term) => term.coefficient === coefficient) ?? false;

export const getSuggestions = async (
  specification: ModelSpecification,
  dataset: Dataset,
): Promise<Suggestion[]> => {
  const activeIds = new Set(specification.alternatives.map((alternative) => alternative.id));
  const costTargets = (["rail", "bus"] as AlternativeId[]).filter(
    (id) => activeIds.has(id) && Boolean(findColumn(dataset, "cost", id)),
  );
  const needsSharedCost =
    activeIds.has("car") &&
    hasTerm(specification, "car", "beta_cost") &&
    costTargets.some((id) => !hasTerm(specification, id, "beta_cost"));
  const referenceId = specification.alternatives.at(-1)?.id;
  const ascTargets = specification.alternatives.filter(
    (alternative) =>
      alternative.id !== referenceId &&
      !hasTerm(specification, alternative.id, `ASC_${alternative.id}`),
  );
  const needsAsc = ascTargets.length > 0;
  const needsIncome =
    activeIds.has("car") &&
    Boolean(findColumn(dataset, "income")) &&
    !hasTerm(specification, "car", "beta_income_car");

  return structuredClone(suggestionCatalog)
    .filter((suggestion) => {
      if (suggestion.id === "share-cost") return needsSharedCost;
      if (suggestion.id === "add-asc") return needsAsc;
      return needsIncome;
    })
    .map((suggestion) => {
      if (suggestion.id === "share-cost") {
        const names = costTargets
          .filter((id) => !hasTerm(specification, id, "beta_cost"))
          .map((id) => dataset.alternatives.find((item) => item.id === id)?.label)
          .filter(Boolean)
          .join(" / ");
        return { ...suggestion, description: `Share beta_cost with ${names}?` };
      }
      if (suggestion.id === "add-asc") {
        const targets = ascTargets.map((item) => item.label).join(" / ");
        const reference = specification.alternatives.at(-1)?.label ?? "the final alternative";
        return {
          ...suggestion,
          description: `Add ${targets} constants while keeping ${reference} as the reference.`,
        };
      }
      const incomeColumn = findColumn(dataset, "income")?.name;
      return {
        ...suggestion,
        description: `Explore whether ${incomeColumn} changes preference for the Car alternative.`,
      };
    });
};

const findColumn = (
  dataset: Dataset,
  concept: "cost" | "income",
  alternativeId?: AlternativeId,
) => dataset.columns.find(
  (column) =>
    column.role === "explanatory" &&
    column.concept === concept &&
    (alternativeId === undefined || column.alternativeId === alternativeId),
);

const newTerm = (
  id: string,
  coefficient: string,
  variable: string,
  kind: UtilityTerm["kind"] = "variable",
): UtilityTerm => ({ id, coefficient, variable, kind });

export const applySuggestion = (
  specification: ModelSpecification,
  suggestionId: SuggestionId,
  dataset: Dataset,
): ModelSpecification => {
  const next = structuredClone(specification);
  const byId = (id: AlternativeId) =>
    next.alternatives.find((alternative) => alternative.id === id);

  if (suggestionId === "share-cost") {
    for (const id of ["rail", "bus"] as AlternativeId[]) {
      const costColumn = findColumn(dataset, "cost", id);
      if (byId(id) && costColumn && !hasTerm(next, id, "beta_cost")) {
        byId(id)?.terms.push(newTerm(`${id}-cost`, "beta_cost", costColumn.name));
      }
    }
  }

  if (suggestionId === "add-asc") {
    const referenceId = next.alternatives.at(-1)?.id;
    for (const alternative of next.alternatives) {
      if (
        alternative.id !== referenceId &&
        !hasTerm(next, alternative.id, `ASC_${alternative.id}`)
      ) {
        alternative.terms.unshift(
          newTerm(`${alternative.id}-asc`, `ASC_${alternative.id}`, "", "constant"),
        );
      }
    }
  }

  const incomeColumn = findColumn(dataset, "income");
  if (
    suggestionId === "income-car" &&
    byId("car") &&
    incomeColumn &&
    !hasTerm(next, "car", "beta_income_car")
  ) {
    byId("car")?.terms.push(
      newTerm("car-income", "beta_income_car", incomeColumn.name),
    );
  }

  return next;
};
