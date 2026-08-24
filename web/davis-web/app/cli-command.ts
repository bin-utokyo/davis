type CliFile = {
  dataset_id: string;
  file_id: string;
};

type CompanionSelection = {
  schema: boolean;
  pdfJa: boolean;
  pdfEn: boolean;
};

export function buildGetCommands(
  files: CliFile[],
  selectedFiles: CliFile[],
  serviceUrl: string,
  companions: CompanionSelection,
) {
  const companionOptions = [
    !companions.schema ? "--no-schema" : null,
    companions.pdfJa ? "--pdf-ja" : null,
    companions.pdfEn ? "--pdf-en" : null,
  ].filter((option): option is string => Boolean(option));
  const serviceOption = `--service-url ${JSON.stringify(serviceUrl)}`;
  const datasetIds = [...new Set(selectedFiles.map((file) => file.dataset_id))];

  return datasetIds.map((id) => {
    const datasetFiles = files.filter((file) => file.dataset_id === id);
    const chosenFiles = selectedFiles.filter((file) => file.dataset_id === id);
    const fileOptions = datasetFiles.length === chosenFiles.length
      ? []
      : chosenFiles.flatMap((file) => ["--file", JSON.stringify(file.file_id)]);
    return ["davis", "get", id, ...fileOptions, ...companionOptions, serviceOption].join(" ");
  }).join("\n");
}
