import assert from "node:assert/strict";
import test from "node:test";

import { buildGetCommands } from "../app/cli-command.ts";

const files = [
  { dataset_id: "routes/sample", file_id: "nodes.csv" },
  { dataset_id: "routes/sample", file_id: "links.csv" },
  { dataset_id: "network/all", file_id: "network.csv" },
];

test("builds CLI commands with the selected companion options", () => {
  const commands = buildGetCommands(
    files,
    [files[0], files[2]],
    "https://davis.example",
    { schema: false, pdfJa: true, pdfEn: true },
  );

  assert.equal(commands, [
    'davis get routes/sample --file "nodes.csv" --no-schema --pdf-ja --pdf-en --service-url "https://davis.example"',
    'davis get network/all --no-schema --pdf-ja --pdf-en --service-url "https://davis.example"',
  ].join("\n"));
});

test("omits companion flags when only the default schema is selected", () => {
  assert.equal(
    buildGetCommands(files, files.slice(0, 2), "https://davis.example", {
      schema: true,
      pdfJa: false,
      pdfEn: false,
    }),
    'davis get routes/sample --service-url "https://davis.example"',
  );
});
