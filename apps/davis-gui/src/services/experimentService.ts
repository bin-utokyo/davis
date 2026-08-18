import { experiments } from "../mock/experiments";
import type { Experiment } from "../types";

export const getExperiments = async (): Promise<Experiment[]> => structuredClone(experiments);
