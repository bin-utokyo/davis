import { createEstimationResult } from "../mock/estimation";
import type { EstimationResult, ModelSpecification } from "../types";

const ESTIMATION_DELAY_MS = 800;

export const runEstimation = async (
  specification: ModelSpecification,
): Promise<EstimationResult> => {
  await new Promise((resolve) => window.setTimeout(resolve, ESTIMATION_DELAY_MS));
  return createEstimationResult(specification);
};
