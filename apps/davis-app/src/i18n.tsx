import { createContext, ReactNode, useContext, useEffect, useState } from "react";

export type Locale = "ja" | "en";

const english: Record<string, string> = {
  "ローカルデータから推定まで": "From local data to estimation",
  "ComponentManifestに従い，入力結合とモデル設定を同じAnalysisPlanとして編集します．": "Edit input joins and model settings in one Analysis Plan, following the Component Manifest.",
  "model.yamlとdavis-runsを置く作業folderです．Davis repositoryのcloneは不要です．": "This workspace stores model.yaml and davis-runs. You do not need to clone the Davis repository.",
  "すべてのcomponentを同じdavis.ui/v1 rendererで編集します．": "Edit every component with the same davis.ui/v1 renderer.",
  "新規Plan": "New Plan", "既存Planを開く": "Open Plan", "空欄ならPlan name": "Uses Plan name when empty",
  "Workspaceを選択してください": "Select a workspace", "このcomponentにはdavis.ui/v1の画面定義がありません．内容を失わないYAML modeで開いています．": "This component has no davis.ui/v1 presentation. It is open in lossless YAML mode.",
  "上書き保存・検証": "Save and validate", "上書きして実行": "Save and run", "YAMLを確認": "Preview YAML",
  "別名で保存": "Save as", "上書き保存": "Save", "上書きして推定": "Save and estimate", "保存して推定": "Save and estimate",
  "結果フォルダを開く": "Open result folder", "Davis Catalogから追加": "Add from Davis Catalog", "閉じる": "Close",
  "データセット名，ファイル名，列名を検索": "Search datasets, files, and columns", "処理中です…": "Working…", "選択": "Choose",
  "入力データ": "Input data", "入力と結合はモデルではなくDavis共通のtable bindingとして保存します．": "Inputs and joins are stored as a shared Davis table binding rather than model-specific settings.",
  "ローカルCSV": "Local CSV", "CSVを1つ以上選択してください．": "Select one or more CSV files.", "削除": "Remove",
  "基準データ": "Base table", "左キー": "Left key", "右キー": "Right key", "未照合を許可": "Allow unmatched rows",
  "列を選択": "Select a column", "parameterと効用へ入れる列を指定します．": "Specify parameters and columns used in utility.",
  "termを追加": "Add term", "効用termを追加してください．": "Add a utility term.", "列": "Column", "定数": "Constant",
  "推定": "Estimate", "固定": "Fixed", "各選択肢を重複なく1つのnestへ入れます．": "Assign each alternative to exactly one nest.",
  "Nestを追加": "Add nest", "2つ以上のnestを追加してください．": "Add at least two nests.", "固定値": "Fixed value",
  "推定初期値": "Initial estimate", "推定または固定": "Estimate or fix", "先に効用termを追加してください．": "Add utility terms first.",
  "任意": "Optional", "自由形式または配列の設定です．": "This setting is a free-form value or array.", "このsectionへ反映": "Apply to this section",
  "検索…": "Search…", "値を検索": "Search values", "候補列を先に指定": "Select the source column first",
  "表": "Table", "指標": "Metrics", "推定parameter": "Parameters", "予測値": "Predictions", "図": "Figure", "診断": "Diagnostics",
  "先頭200行を表示しています．": "Showing the first 200 rows.",
  "davis.ui/v1に対応するcomponentが見つかりません．": "No component with a davis.ui/v1 presentation was found.",
  "利用できるcomponentがありません．先にDavis CLIで公式componentをインストールしてください．": "No components are available. Install the official components with Davis CLI first.",
  "componentがまだインストールされていません．": "No components are installed yet.",
  "Project workspaceは正しく選択されています．ターミナルで公式componentをインストールしてから，Workspaceを選択し直してください．": "Your project workspace is valid. Install the official components in a terminal, then select the workspace again.",
  "このcomponentはdavis.ui/v1に対応していません．": "This component does not support davis.ui/v1.",
  "{id}の入力データを選択してください．": "Select input data for {id}.", "{slot}の{source}について結合キーを選択してください．": "Select join keys for {source} in {slot}.",
  "先にWorkspaceを選択してください．": "Select a workspace first.", "読み込めませんでした": "Could not load",
  "GUIで扱えない入力形式です．": "This input form cannot be edited in the GUI.", "保存先がありません．既存Planを開き直してください．": "No save destination is available. Reopen the existing Plan.",
  "として保存・検証しました．": "was saved and validated.", "生成されたmodel.yaml": "Generated model.yaml",
  "へ追加するファイルを選択すると，自動で共有データ領域へダウンロードします．": "Select a file to add; Davis will download it to the shared data area.",
  "{id}の入力ファイルを解決できません．": "Could not resolve the input file for {id}.", "{id}はGUIで扱えない入力形式です．": "Input {id} cannot be edited in the GUI.",
  "UI extension {id}を読み込めません．": "Could not load UI extension {id}.", "widget {id}はこのDesktopに未実装です．": "Widget {id} is not implemented by this Desktop version.",
  "候補値": "Candidate values", "件": "values", "取得行数": "sampled rows", "上限付きsample": "bounded sample", "行": "rows",
  "入力 {id} が未選択です．": "Input {id} is not selected.", "列参照 {id} が未設定です．": "Column reference {id} is not set.",
  "列 {column} を入力 {input} から解決できません．": "Could not resolve column {column} from input {input}.",
  "としてAnalysis Planへ保存します．": " is saved in the Analysis Plan.", "このsectionだけYAMLで編集できます．": "Only this section can be edited as YAML.",
  "過去の実行を開き，標準artifact profileを使って比較します．": "Open previous runs and compare them using standard artifact profiles.",
  "更新": "Refresh", "選択したRunを比較": "Compare selected runs", "このWorkspaceにはまだRunがありません．": "This workspace has no runs yet.",
  "読み込めなかったRunがあります": "Some runs could not be loaded", "比較": "Compare", "結果を見る": "View results",
  "artifact名ではなくManifestに記録されたmetrics・parameters profileを照合しています．": "Comparison uses the metrics and parameters profiles declared by the Manifest rather than artifact filenames.",
  "選択したRunに比較可能なJSON／CSVの標準artifactがありません．": "The selected runs have no comparable standard JSON or CSV artifacts.",
};

type I18nContextValue = { locale: Locale; setLocale: (locale: Locale) => void; t: (japanese: string) => string };
const I18nContext = createContext<I18nContextValue>({ locale: "ja", setLocale: () => {}, t: (value) => value });

function initialLocale(): Locale {
  const saved = localStorage.getItem("davis.locale");
  if (saved === "ja" || saved === "en") return saved;
  return navigator.language.toLowerCase().startsWith("ja") ? "ja" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>(initialLocale);
  useEffect(() => { localStorage.setItem("davis.locale", locale); document.documentElement.lang = locale; }, [locale]);
  return <I18nContext.Provider value={{ locale, setLocale, t: (value) => locale === "en" ? english[value] ?? value : value }}>{children}</I18nContext.Provider>;
}

export function useI18n() { return useContext(I18nContext); }

export function localizedText(value: unknown, locale: Locale, fallback = ""): string {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) return fallback;
  const text = value as Record<string, unknown>;
  for (const key of [locale, locale === "ja" ? "en" : "ja"]) if (typeof text[key] === "string") return text[key] as string;
  return fallback;
}

export function localizeTree<T>(value: T, locale: Locale): T {
  if (Array.isArray(value)) return value.map((item) => localizeTree(item, locale)) as T;
  if (!value || typeof value !== "object") return value;
  const record = value as Record<string, unknown>;
  if ((typeof record.ja === "string" || typeof record.en === "string") && Object.keys(record).every((key) => key === "ja" || key === "en")) return localizedText(record, locale) as T;
  const localized = Object.fromEntries(Object.entries(record).map(([key, item]) => [key, localizeTree(item, locale)])) as Record<string, unknown>;
  if (record["x-davis-title"]) localized.title = localizedText(record["x-davis-title"], locale, typeof record.title === "string" ? record.title : "");
  if (record["x-davis-description"]) localized.description = localizedText(record["x-davis-description"], locale, typeof record.description === "string" ? record.description : "");
  return localized as T;
}
