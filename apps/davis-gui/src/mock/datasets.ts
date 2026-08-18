import type { Dataset } from "../types";

const alternatives = [
  { id: "car", label: "CAR", color: "#756bc3" },
  { id: "rail", label: "RAIL", color: "#60a5fa" },
  { id: "bus", label: "BUS", color: "#34d399" },
  { id: "walk", label: "WALK", color: "#fbbf24" },
] as const;

export const datasets: Dataset[] = [
  {
    id: "tokyo-pt-2018",
    name: "Tokyo PT 2018",
    observations: 43_812,
    alternatives: [...alternatives],
    columns: [
      { name: "trip_id", label: "Trip ID", type: "id", role: "id" },
      { name: "choice", label: "Chosen mode", type: "category", role: "choice" },
      { name: "car_time", label: "Car travel time", type: "number", role: "explanatory", alternativeId: "car", concept: "time" },
      { name: "car_cost", label: "Car cost", type: "number", role: "explanatory", alternativeId: "car", concept: "cost" },
      { name: "rail_time", label: "Rail travel time", type: "number", role: "explanatory", alternativeId: "rail", concept: "time" },
      { name: "rail_cost", label: "Rail cost", type: "number", role: "explanatory", alternativeId: "rail", concept: "cost" },
      { name: "bus_time", label: "Bus travel time", type: "number", role: "explanatory", alternativeId: "bus", concept: "time" },
      { name: "bus_cost", label: "Bus cost", type: "number", role: "explanatory", alternativeId: "bus", concept: "cost" },
      { name: "walk_time", label: "Walk travel time", type: "number", role: "explanatory", alternativeId: "walk", concept: "time" },
      { name: "income", label: "Household income", type: "number", role: "explanatory", concept: "income" },
    ],
    description: "Metropolitan person-trip survey",
  },
  {
    id: "synthetic-commute",
    name: "Synthetic Commute",
    observations: 12_400,
    alternatives: [...alternatives],
    columns: [
      { name: "record_id", label: "Record ID", type: "id", role: "id" },
      { name: "chosen_mode", label: "Chosen mode", type: "category", role: "choice" },
      { name: "drive_minutes", label: "Driving minutes", type: "number", role: "explanatory", alternativeId: "car", concept: "time" },
      { name: "drive_fare", label: "Driving cost", type: "number", role: "explanatory", alternativeId: "car", concept: "cost" },
      { name: "rail_minutes", label: "Rail minutes", type: "number", role: "explanatory", alternativeId: "rail", concept: "time" },
      { name: "rail_fare", label: "Rail fare", type: "number", role: "explanatory", alternativeId: "rail", concept: "cost" },
      { name: "bus_minutes", label: "Bus minutes", type: "number", role: "explanatory", alternativeId: "bus", concept: "time" },
      { name: "bus_fare", label: "Bus fare", type: "number", role: "explanatory", alternativeId: "bus", concept: "cost" },
      { name: "walk_minutes", label: "Walk minutes", type: "number", role: "explanatory", alternativeId: "walk", concept: "time" },
      { name: "household_income", label: "Household income", type: "number", role: "explanatory", concept: "income" },
    ],
    description: "Calibrated commuting choices",
  },
  {
    id: "teaching-sample",
    name: "Teaching Sample",
    observations: 2_500,
    alternatives: [...alternatives],
    columns: [
      { name: "case_id", label: "Case ID", type: "id", role: "id" },
      { name: "mode_choice", label: "Mode choice", type: "category", role: "choice" },
      { name: "time_car", label: "Car time", type: "number", role: "explanatory", alternativeId: "car", concept: "time" },
      { name: "cost_car", label: "Car cost", type: "number", role: "explanatory", alternativeId: "car", concept: "cost" },
      { name: "time_rail", label: "Rail time", type: "number", role: "explanatory", alternativeId: "rail", concept: "time" },
      { name: "cost_rail", label: "Rail cost", type: "number", role: "explanatory", alternativeId: "rail", concept: "cost" },
      { name: "time_bus", label: "Bus time", type: "number", role: "explanatory", alternativeId: "bus", concept: "time" },
      { name: "cost_bus", label: "Bus cost", type: "number", role: "explanatory", alternativeId: "bus", concept: "cost" },
      { name: "time_walk", label: "Walk time", type: "number", role: "explanatory", alternativeId: "walk", concept: "time" },
      { name: "income_monthly", label: "Monthly income", type: "number", role: "explanatory", concept: "income" },
    ],
    description: "Compact classroom dataset",
  },
];
