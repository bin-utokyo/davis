import type { Suggestion } from "../types";

export const suggestionCatalog: Suggestion[] = [
  {
    id: "share-cost",
    eyebrow: "Generic coefficient",
    title: "Cost appears only in Car.",
    description: "Share beta_cost across Car / Rail / Bus?",
    impact: "High",
  },
  {
    id: "add-asc",
    eyebrow: "Specification",
    title: "Consider adding alternative-specific constants.",
    description: "Add Rail and Bus constants while keeping Walk as the reference.",
    impact: "Medium",
  },
  {
    id: "income-car",
    eyebrow: "Interaction",
    title: "Test income × Car interaction.",
    description: "Explore whether income changes preference for the Car alternative.",
    impact: "Explore",
  },
];
