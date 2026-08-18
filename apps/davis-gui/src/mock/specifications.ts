import type { AlternativeId, ModelSpecification, UtilityTerm } from "../types";

const term = (
  id: string,
  coefficient: string,
  variable = "",
  kind: UtilityTerm["kind"] = "variable",
): UtilityTerm => ({ id, coefficient, variable, kind });

const alternative = (
  id: AlternativeId,
  label: string,
  color: string,
  terms: UtilityTerm[],
) => ({ id, label, color, terms });

export const timeOnlySpecification: ModelSpecification = {
  alternatives: [
    alternative("car", "CAR", "#a78bfa", [term("car-time", "beta_time", "car_time")]),
    alternative("rail", "RAIL", "#60a5fa", [term("rail-time", "beta_time", "rail_time")]),
    alternative("bus", "BUS", "#34d399", [term("bus-time", "beta_time", "bus_time")]),
    alternative("walk", "WALK", "#fbbf24", [term("walk-time", "beta_time", "walk_time")]),
  ],
};

export const initialSpecification: ModelSpecification = {
  alternatives: [
    alternative("car", "CAR", "#a78bfa", [
      term("car-asc", "ASC_car", "", "constant"),
      term("car-time", "beta_time", "car_time"),
      term("car-cost", "beta_cost", "car_cost"),
    ]),
    alternative("rail", "RAIL", "#60a5fa", [term("rail-time", "beta_time", "rail_time")]),
    alternative("bus", "BUS", "#34d399", [term("bus-time", "beta_time", "bus_time")]),
    alternative("walk", "WALK", "#fbbf24", [term("walk-time", "beta_time", "walk_time")]),
  ],
};

export const timeCostSpecification: ModelSpecification = {
  alternatives: [
    alternative("car", "CAR", "#a78bfa", [
      term("car-time", "beta_time", "car_time"),
      term("car-cost", "beta_cost", "car_cost"),
    ]),
    alternative("rail", "RAIL", "#60a5fa", [
      term("rail-time", "beta_time", "rail_time"),
      term("rail-cost", "beta_cost", "rail_cost"),
    ]),
    alternative("bus", "BUS", "#34d399", [
      term("bus-time", "beta_time", "bus_time"),
      term("bus-cost", "beta_cost", "bus_cost"),
    ]),
    alternative("walk", "WALK", "#fbbf24", [term("walk-time", "beta_time", "walk_time")]),
  ],
};

export const fullSpecification: ModelSpecification = {
  alternatives: [
    alternative("car", "CAR", "#a78bfa", [
      term("car-asc", "ASC_car", "", "constant"),
      term("car-time", "beta_time", "car_time"),
      term("car-cost", "beta_cost", "car_cost"),
    ]),
    alternative("rail", "RAIL", "#60a5fa", [
      term("rail-asc", "ASC_rail", "", "constant"),
      term("rail-time", "beta_time", "rail_time"),
      term("rail-cost", "beta_cost", "rail_cost"),
    ]),
    alternative("bus", "BUS", "#34d399", [
      term("bus-asc", "ASC_bus", "", "constant"),
      term("bus-time", "beta_time", "bus_time"),
      term("bus-cost", "beta_cost", "bus_cost"),
    ]),
    alternative("walk", "WALK", "#fbbf24", [term("walk-time", "beta_time", "walk_time")]),
  ],
};

export const cloneSpecification = (specification: ModelSpecification): ModelSpecification =>
  structuredClone(specification);
