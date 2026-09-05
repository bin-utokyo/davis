import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { useState } from "react";

type ColumnProfile = {
  name: string;
  inferred_type: string;
  null_count: number;
  unique_sample: number;
  warnings: string[];
};

type CsvProfile = {
  path: string;
  encoding: string;
  delimiter: string;
  rows_sampled: number;
  truncated: boolean;
  columns: ColumnProfile[];
};

type Validation = {
  valid: boolean;
  plan: string;
  component: { id: string; version: string; manifest: string };
};

type CompletedRun = {
  run_directory: string;
  request: { run_id: string; component: { id: string; version: string } };
  result: {
    status: string;
    artifacts: Record<string, { path: string; media_type: string; size?: number }>;
    extensions: Record<string, { path: string; media_type: string; size?: number }>;
  };
};

export default function App() {
  const [repository, setRepository] = useState("");
  const [csvPath, setCsvPath] = useState("");
  const [planPath, setPlanPath] = useState("");
  const [profile, setProfile] = useState<CsvProfile>();
  const [validation, setValidation] = useState<Validation>();
  const [completed, setCompleted] = useState<CompletedRun>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function chooseRepository() {
    const selected = await open({ directory: true, multiple: false });
    if (typeof selected === "string") setRepository(selected);
  }

  async function chooseCsv() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "CSV", extensions: ["csv", "tsv"] }],
    });
    if (typeof selected !== "string") return;
    setCsvPath(selected);
    await perform(async () => {
      setProfile(await invoke<CsvProfile>("inspect_csv_file", { path: selected }));
    });
  }

  async function choosePlan() {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Davis analysis plan", extensions: ["yaml", "yml"] }],
    });
    if (typeof selected === "string") setPlanPath(selected);
  }

  async function validate() {
    await perform(async () => {
      setValidation(
        await invoke<Validation>("validate_analysis_plan", {
          repository,
          plan: planPath,
        }),
      );
      setCompleted(undefined);
    });
  }

  async function run() {
    await perform(async () => {
      setCompleted(
        await invoke<CompletedRun>("run_analysis_plan", {
          repository,
          plan: planPath,
        }),
      );
    });
  }

  async function openRunDirectory() {
    if (!completed) return;
    await perform(async () => {
      await invoke("open_run_directory", {
        repository,
        runId: completed.request.run_id,
      });
    });
  }

  async function perform(action: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action();
    } catch (reason) {
      setError(String(reason));
    } finally {
      setBusy(false);
    }
  }

  const ready = repository.length > 0 && planPath.length > 0;
  return (
    <main>
      <header>
        <p className="eyebrow">DAVIS MODEL</p>
        <h1>ローカルデータから推定まで</h1>
        <p className="lead">
          データを外部へ送信せず，Davis共通protocolでCSVを確認し，model componentを実行します．
        </p>
      </header>

      {error && <div className="error">{error}</div>}

      <section>
        <div className="section-heading">
          <span>1</span>
          <div>
            <h2>Workspace</h2>
            <p>componentとrun記録を置くDavis repositoryを選択します．</p>
          </div>
        </div>
        <PathField value={repository} placeholder="Davis repository" onChange={setRepository} onChoose={chooseRepository} />
      </section>

      <section>
        <div className="section-heading">
          <span>2</span>
          <div>
            <h2>CSV inspection</h2>
            <p>encoding，区切り文字，型推論，先頭0を確認します．</p>
          </div>
        </div>
        <PathField value={csvPath} placeholder="CSV file" onChange={setCsvPath} onChoose={chooseCsv} />
        {profile && <ProfileTable profile={profile} />}
      </section>

      <section>
        <div className="section-heading">
          <span>3</span>
          <div>
            <h2>Analysis plan</h2>
            <p>GUIとAIが共通利用するmodel.yamlを検証して実行します．</p>
          </div>
        </div>
        <PathField value={planPath} placeholder="model.yaml" onChange={setPlanPath} onChoose={choosePlan} />
        <div className="actions">
          <button className="secondary" disabled={!ready || busy} onClick={validate}>検証する</button>
          <button disabled={!ready || busy} onClick={run}>推定を実行する</button>
        </div>
        {validation && (
          <div className="success">
            {validation.component.id} {validation.component.version}を実行できます．
          </div>
        )}
      </section>

      {completed && (
        <section>
          <div className="section-heading">
            <span>4</span>
            <div>
              <h2>Run result</h2>
              <p>{completed.request.run_id}</p>
            </div>
          </div>
          <div className="run-directory-row">
            <div className="run-directory">{completed.run_directory}</div>
            <button className="secondary" disabled={busy} onClick={openRunDirectory}>結果フォルダを開く</button>
          </div>
          <div className="artifacts">
            {[
              ...Object.entries(completed.result.artifacts),
              ...Object.entries(completed.result.extensions),
            ].map(([name, artifact]) => (
              <article key={name}>
                <strong>{name}</strong>
                <span>{artifact.path}</span>
                <small>{artifact.media_type}{artifact.size ? ` · ${artifact.size} bytes` : ""}</small>
              </article>
            ))}
          </div>
        </section>
      )}

      {busy && <div className="busy">処理中です…</div>}
    </main>
  );
}

function PathField({
  value,
  placeholder,
  onChange,
  onChoose,
}: {
  value: string;
  placeholder: string;
  onChange: (value: string) => void;
  onChoose: () => void;
}) {
  return (
    <div className="path-field">
      <input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
      <button className="secondary" onClick={onChoose}>選択</button>
    </div>
  );
}

function ProfileTable({ profile }: { profile: CsvProfile }) {
  return (
    <div className="profile">
      <div className="profile-summary">
        <span>{profile.encoding}</span><span>{JSON.stringify(profile.delimiter)}区切り</span><span>{profile.rows_sampled}行確認</span>
      </div>
      <table>
        <thead><tr><th>列</th><th>推論型</th><th>欠損</th><th>異なる値</th><th>警告</th></tr></thead>
        <tbody>
          {profile.columns.map((column) => (
            <tr key={column.name}>
              <td>{column.name}</td><td>{column.inferred_type}</td><td>{column.null_count}</td>
              <td>{column.unique_sample}</td><td>{column.warnings.join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
