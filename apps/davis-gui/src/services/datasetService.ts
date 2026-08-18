import { datasets } from "../mock/datasets";
import type { Dataset } from "../types";

export const getDatasets = async (): Promise<Dataset[]> => structuredClone(datasets);
