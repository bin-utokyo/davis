import type { Experiment } from "../types";
import { createEstimationResult } from "./estimation";
import {
  cloneSpecification,
  fullSpecification,
  timeCostSpecification,
  timeOnlySpecification,
} from "./specifications";

export const experiments: Experiment[] = [
  {
    id: "mnl-03",
    datasetId: "tokyo-pt-2018",
    name: "MNL-03",
    summary: "Time + Cost + ASC",
    createdAt: "18 minutes ago",
    specification: cloneSpecification(fullSpecification),
    result: createEstimationResult(fullSpecification),
  },
  {
    id: "mnl-02",
    datasetId: "tokyo-pt-2018",
    name: "MNL-02",
    summary: "Time + Cost",
    createdAt: "Yesterday",
    specification: cloneSpecification(timeCostSpecification),
    result: createEstimationResult(timeCostSpecification),
  },
  {
    id: "mnl-01",
    datasetId: "tokyo-pt-2018",
    name: "MNL-01",
    summary: "Time only",
    createdAt: "3 days ago",
    specification: cloneSpecification(timeOnlySpecification),
    result: createEstimationResult(timeOnlySpecification),
  },
];
