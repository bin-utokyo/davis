import { suggestionCatalog } from "../mock/suggestions";
import type {
  AlternativeId,
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
): Promise<Suggestion[]> => {
  const needsSharedCost =
    hasTerm(specification, "car", "beta_cost") &&
    (!hasTerm(specification, "rail", "beta_cost") ||
      !hasTerm(specification, "bus", "beta_cost"));
  const needsAsc =
    !hasTerm(specification, "rail", "ASC_rail") ||
    !hasTerm(specification, "bus", "ASC_bus");
  const needsIncome = !hasTerm(specification, "car", "beta_income_car");

  return structuredClone(
    suggestionCatalog.filter((suggestion) => {
      if (suggestion.id === "share-cost") return needsSharedCost;
      if (suggestion.id === "add-asc") return needsAsc;
      return needsIncome;
    }),
  );
};

const newTerm = (
  id: string,
  coefficient: string,
  variable: string,
  kind: UtilityTerm["kind"] = "variable",
): UtilityTerm => ({ id, coefficient, variable, kind });

export const applySuggestion = (
  specification: ModelSpecification,
  suggestionId: SuggestionId,
): ModelSpecification => {
  const next = structuredClone(specification);
  const byId = (id: AlternativeId) =>
    next.alternatives.find((alternative) => alternative.id === id);

  if (suggestionId === "share-cost") {
    if (!hasTerm(next, "rail", "beta_cost")) {
      byId("rail")?.terms.push(newTerm("rail-cost", "beta_cost", "rail_cost"));
    }
    if (!hasTerm(next, "bus", "beta_cost")) {
      byId("bus")?.terms.push(newTerm("bus-cost", "beta_cost", "bus_cost"));
    }
  }

  if (suggestionId === "add-asc") {
    if (!hasTerm(next, "rail", "ASC_rail")) {
      byId("rail")?.terms.unshift(newTerm("rail-asc", "ASC_rail", "", "constant"));
    }
    if (!hasTerm(next, "bus", "ASC_bus")) {
      byId("bus")?.terms.unshift(newTerm("bus-asc", "ASC_bus", "", "constant"));
    }
  }

  if (suggestionId === "income-car" && !hasTerm(next, "car", "beta_income_car")) {
    byId("car")?.terms.push(newTerm("car-income", "beta_income_car", "income"));
  }

  return next;
};
