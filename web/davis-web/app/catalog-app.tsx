"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { buildGetCommands } from "./cli-command";

type LocalizedText = { ja: string; en: string };
type Language = "ja" | "en";
type Column = { name: string; data_type: string; description: LocalizedText | null };
type CatalogDocument = { id: string; path: string; size: number };
type ExternalDocument = { id: string; path: string; size: number; url: string };
type CatalogFile = {
  id: string;
  dataset_id: string;
  file_id: string;
  path: string;
  size: number;
  object: { oid: string; size: number };
  format: string;
  schema_status: "ready" | "missing" | "invalid";
  schema_path: string | null;
  schema_error: string | null;
  name: LocalizedText | null;
  description: LocalizedText | null;
  city: LocalizedText | null;
  year: number | null;
  license: LocalizedText | null;
  columns: Column[];
  raw_schema: string | null;
  documents?: {
    schema?: CatalogDocument | null;
    pdf_ja?: ExternalDocument | null;
    pdf_en?: ExternalDocument | null;
  };
};
type Dataset = {
  id: string;
  root: string;
  file_count: number;
  schema_ready_count: number;
  total_size: number;
};
type Facets = {
  cities: LocalizedText[];
  years: number[];
  formats: string[];
  licenses: LocalizedText[];
  schema_statuses: Array<"ready" | "missing" | "invalid">;
};
type SessionState = "checking" | "authenticated" | "anonymous";
type DownloadGrant = { file_id: string; path: string; size: number; expires_at: string; url: string };

const emptyFacets: Facets = { cities: [], years: [], formats: [], licenses: [], schema_statuses: [] };

function searchable(file: CatalogFile) {
  const columnText = file.columns.flatMap((column) => [
    column.name,
    column.data_type,
    column.description?.ja,
    column.description?.en,
  ]);
  return [
    file.dataset_id,
    file.file_id,
    file.path,
    file.name?.ja,
    file.name?.en,
    file.description?.ja,
    file.description?.en,
    file.city?.ja,
    file.city?.en,
    file.year,
    file.license?.ja,
    file.license?.en,
    ...columnText,
  ].filter(Boolean).join(" ").toLocaleLowerCase();
}

function humanSize(bytes: number) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let size = bytes;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function datasetLabel(id: string) {
  return id.split("/").at(-1)?.replaceAll("-", " ") ?? id;
}

function localized(value: LocalizedText | null, language: Language) {
  if (!value) return "";
  return value[language]?.trim() || value.ja?.trim() || value.en?.trim() || "";
}

export function CatalogApp() {
  const [language, setLanguage] = useState<Language>("ja");
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [files, setFiles] = useState<CatalogFile[]>([]);
  const [facets, setFacets] = useState<Facets>(emptyFacets);
  const [loadingError, setLoadingError] = useState(false);
  const [query, setQuery] = useState("");
  const [city, setCity] = useState("");
  const [year, setYear] = useState("");
  const [format, setFormat] = useState("");
  const [schemaStatus, setSchemaStatus] = useState("");
  const [license, setLicense] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(true);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeDataset, setActiveDataset] = useState<string | null>(null);
  const [activeFile, setActiveFile] = useState<CatalogFile | null>(null);
  const [copied, setCopied] = useState(false);
  const [sessionState, setSessionState] = useState<SessionState>("checking");
  const [sessionExpiresAt, setSessionExpiresAt] = useState("");
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false);
  const [inviteCode, setInviteCode] = useState("");
  const [licenseConfirmed, setLicenseConfirmed] = useState(false);
  const [authPending, setAuthPending] = useState(false);
  const [downloadPending, setDownloadPending] = useState(false);
  const [accessError, setAccessError] = useState<"login" | "session" | "download" | "">("");
  const [downloadCount, setDownloadCount] = useState(0);
  const [includeSchema, setIncludeSchema] = useState(true);
  const [includePdfJa, setIncludePdfJa] = useState(false);
  const [includePdfEn, setIncludePdfEn] = useState(false);
  const tr = (ja: string, en: string) => language === "ja" ? ja : en;

  useEffect(() => {
    const stored = window.localStorage.getItem("davis-language");
    if (stored !== "ja" && stored !== "en") return;
    const timeout = window.setTimeout(() => setLanguage(stored), 0);
    return () => window.clearTimeout(timeout);
  }, []);

  useEffect(() => {
    document.documentElement.lang = language;
    document.title = language === "ja" ? "Davis | 交通データカタログ" : "Davis | Transport Data Catalog";
    window.localStorage.setItem("davis-language", language);
  }, [language]);

  useEffect(() => {
    Promise.all([
      fetch("/catalog/datasets.json").then((response) => response.json() as Promise<Dataset[]>),
      fetch("/catalog/files.json").then((response) => response.json() as Promise<CatalogFile[]>),
      fetch("/catalog/facets.json").then((response) => response.json() as Promise<Facets>),
    ]).then(([nextDatasets, nextFiles, nextFacets]) => {
      setDatasets(nextDatasets);
      setFiles(nextFiles);
      setFacets(nextFacets);
    }).catch(() => setLoadingError(true));
  }, []);

  useEffect(() => {
    fetch("/api/v1/auth/session", { credentials: "same-origin" }).then(async (response) => {
      if (!response.ok) {
        setSessionState("anonymous");
        return;
      }
      const body = await response.json() as { expires_at: string };
      setSessionExpiresAt(body.expires_at);
      setSessionState("authenticated");
    }).catch(() => setSessionState("anonymous"));
  }, []);

  const normalizedQuery = query.trim().toLocaleLowerCase();
  const filteredFiles = useMemo(() => files.filter((file) => {
    if (normalizedQuery && !searchable(file).includes(normalizedQuery)) return false;
    if (city && file.city?.ja !== city) return false;
    if (year && file.year !== Number(year)) return false;
    if (format && file.format !== format) return false;
    if (schemaStatus && file.schema_status !== schemaStatus) return false;
    if (license && file.license?.ja !== license) return false;
    return true;
  }), [files, normalizedQuery, city, year, format, schemaStatus, license]);

  const grouped = useMemo(() => {
    const groups = new Map<string, CatalogFile[]>();
    for (const file of filteredFiles) {
      const group = groups.get(file.dataset_id) ?? [];
      group.push(file);
      groups.set(file.dataset_id, group);
    }
    return groups;
  }, [filteredFiles]);

  const visibleDatasets = datasets.filter((dataset) => grouped.has(dataset.id));
  const activeDatasetFiles = activeDataset ? files.filter((file) => file.dataset_id === activeDataset) : [];
  const selectedFiles = files.filter((file) => selected.has(file.id));
  const selectedSize = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const selectedHasSchema = selectedFiles.some((file) => Boolean(file.documents?.schema));
  const selectedHasPdfJa = selectedFiles.some((file) => Boolean(file.documents?.pdf_ja));
  const selectedHasPdfEn = selectedFiles.some((file) => Boolean(file.documents?.pdf_en));
  const shouldIncludeSchema = selectedHasSchema && includeSchema;
  const shouldIncludePdfJa = selectedHasPdfJa && includePdfJa;
  const shouldIncludePdfEn = selectedHasPdfEn && includePdfEn;
  const selectedSchemaDocuments = selectedFiles.flatMap((file) => [
    shouldIncludeSchema ? file.documents?.schema : null,
  ].filter((document): document is CatalogDocument => Boolean(document))
    .map((document) => ({ ...document, contents: file.raw_schema ?? "" })));
  const selectedPdfDocuments = selectedFiles.flatMap((file) => [
    shouldIncludePdfJa ? file.documents?.pdf_ja : null,
    shouldIncludePdfEn ? file.documents?.pdf_en : null,
  ].filter((document): document is ExternalDocument => Boolean(document)));
  const selectedDownloadIds = [
    ...selectedFiles.map((file) => file.id),
  ];
  const selectedDownloadCount = selectedDownloadIds.length + selectedPdfDocuments.length;
  const selectedDownloadSize = selectedSize
    + selectedSchemaDocuments.reduce((sum, document) => sum + document.size, 0)
    + selectedPdfDocuments.reduce((sum, document) => sum + document.size, 0);

  function toggleFile(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  }

  function toggleDataset(datasetId: string) {
    const ids = files.filter((file) => file.dataset_id === datasetId).map((file) => file.id);
    const allSelected = ids.every((id) => selected.has(id));
    setSelected((current) => {
      const next = new Set(current);
      for (const id of ids) {
        if (allSelected) next.delete(id); else next.add(id);
      }
      return next;
    });
  }

  function applyQuickSearch(value: string) {
    setQuery(value);
    document.querySelector("#catalog")?.scrollIntoView({ behavior: "smooth" });
  }

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    document.querySelector("#catalog")?.scrollIntoView({ behavior: "smooth" });
  }

  async function copyCommands() {
    const commands = buildGetCommands(files, selectedFiles, window.location.origin, {
      schema: shouldIncludeSchema,
      pdfJa: shouldIncludePdfJa,
      pdfEn: shouldIncludePdfEn,
    });
    await navigator.clipboard.writeText(commands);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function openDownloadDialog() {
    setAccessError("");
    setDownloadCount(0);
    setLicenseConfirmed(false);
    setDownloadDialogOpen(true);
  }

  async function login(event: FormEvent) {
    event.preventDefault();
    setAuthPending(true);
    setAccessError("");
    try {
      const response = await fetch("/api/v1/auth/exchange", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ invite_code: inviteCode, client: "web" }),
      });
      if (!response.ok) throw new Error("login");
      const body = await response.json() as { expires_at: string };
      setInviteCode("");
      setSessionExpiresAt(body.expires_at);
      setSessionState("authenticated");
    } catch {
      setAccessError("login");
    } finally {
      setAuthPending(false);
    }
  }

  async function logout() {
    await fetch("/api/v1/auth/logout", { method: "POST", credentials: "same-origin" }).catch(() => null);
    setSessionState("anonymous");
    setSessionExpiresAt("");
  }

  async function downloadSelected() {
    if (!licenseConfirmed || selectedFiles.length === 0 || sessionState !== "authenticated") return;
    setDownloadPending(true);
    setAccessError("");
    setDownloadCount(0);
    try {
      const grants: DownloadGrant[] = [];
      for (let offset = 0; offset < selectedDownloadIds.length; offset += 256) {
        const response = await fetch("/api/v1/download-grants", {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_ids: selectedDownloadIds.slice(offset, offset + 256) }),
        });
        if (response.status === 401) {
          setSessionState("anonymous");
          throw new Error("session");
        }
        if (!response.ok) throw new Error("download");
        const body = await response.json() as { grants: DownloadGrant[] };
        grants.push(...body.grants);
      }
      for (const grant of grants) {
        const link = document.createElement("a");
        link.href = grant.url;
        link.download = grant.path.split("/").at(-1) ?? "download";
        link.rel = "noopener";
        document.body.append(link);
        link.click();
        link.remove();
        await new Promise((resolveDelay) => window.setTimeout(resolveDelay, 120));
      }
      for (const schemaDocument of selectedSchemaDocuments) {
        const url = URL.createObjectURL(new Blob([schemaDocument.contents], { type: "application/yaml;charset=utf-8" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = schemaDocument.path.split("/").at(-1) ?? "schema.yaml";
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        await new Promise((resolveDelay) => window.setTimeout(resolveDelay, 120));
      }
      for (const pdfDocument of selectedPdfDocuments) {
        const response = await fetch(pdfDocument.url);
        if (!response.ok) throw new Error("download");
        const url = URL.createObjectURL(await response.blob());
        const link = document.createElement("a");
        link.href = url;
        link.download = pdfDocument.path.split("/").at(-1) ?? "document.pdf";
        link.rel = "noopener";
        document.body.append(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
        await new Promise((resolveDelay) => window.setTimeout(resolveDelay, 120));
      }
      setDownloadCount(grants.length + selectedSchemaDocuments.length + selectedPdfDocuments.length);
    } catch (error) {
      setAccessError(error instanceof Error && error.message === "session" ? "session" : "download");
    } finally {
      setDownloadPending(false);
    }
  }

  const selectedLicenses = [...new Set(selectedFiles.map((file) => localized(file.license, language)).filter(Boolean))];

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Davis catalog home"><span className="brand-mark">D</span><span>Davis</span></a>
        <nav aria-label={tr("メインナビゲーション", "Main navigation")}><a className="active" href="#catalog">{tr("データを探す", "Find data")}</a><a href="#guide">{tr("使い方", "How to use")}</a><a href="#about">{tr("Davisについて", "About Davis")}</a></nav>
        <div className="header-actions">
          <div className="language-switch" role="group" aria-label={tr("表示言語", "Display language")}><button type="button" aria-pressed={language === "ja"} onClick={() => setLanguage("ja")}>日本語</button><button type="button" aria-pressed={language === "en"} onClick={() => setLanguage("en")}>English</button></div>
          {sessionState === "authenticated" ? <button className="auth-button" type="button" onClick={logout}>{tr("ログアウト", "Log out")}</button> : <button className="auth-button" type="button" onClick={openDownloadDialog}>{sessionState === "checking" ? tr("確認中", "Checking") : tr("ログイン", "Log in")}</button>}
          <button className="selection-button" type="button" onClick={() => selected.size && document.querySelector("#selection")?.scrollIntoView()}>
            {tr("選択中", "Selected")} <span>{selected.size}</span>
          </button>
        </div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow">TRANSPORT DATA CATALOG</div>
        <h1>{language === "ja" ? <>交通データを，<br />研究のすぐそばに．</> : <>Transport data,<br />ready for your research.</>}</h1>
        <p>{tr("データの内容と列定義を確かめながら，必要なファイルをまとめて探せる交通行動研究のためのカタログです．", "A catalog for travel behavior research that lets you inspect data descriptions and column definitions while finding the files you need.")}</p>
        <form className="search-panel" role="search" onSubmit={submitSearch}>
          <label htmlFor="catalog-search">{tr("データを検索", "Search data")}</label>
          <div className="search-row"><span aria-hidden="true">⌕</span><input id="catalog-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr("地域，年，列名，データの説明から検索", "Search by location, year, column, or description")}/><button type="submit">{tr("検索", "Search")}</button></div>
          <div className="quick-filters" aria-label={tr("検索例", "Search examples")}><span>{tr("検索例", "Examples")}</span>{(language === "ja" ? ["松山", "2021", "travel_time"] : ["Matsuyama", "2021", "travel_time"]).map((value) => <button key={value} type="button" onClick={() => applyQuickSearch(value)}>{value}</button>)}</div>
        </form>
        <div className="hero-stats" aria-label={tr("カタログ概要", "Catalog overview")}><div><strong>{datasets.length || "–"}</strong><span>datasets</span></div><div><strong>{files.length || "–"}</strong><span>files</span></div><div><strong>{files.filter((file) => file.schema_status === "ready").length || "–"}</strong><span>schemas</span></div></div>
      </section>

      <section className="catalog-section" id="catalog">
        <div className="section-heading"><div><p className="section-kicker">CATALOG</p><h2>{tr("データセットを探す", "Find datasets")}</h2></div><button className="filter-button" type="button" aria-expanded={filtersOpen} onClick={() => setFiltersOpen((value) => !value)}>{tr("絞り込み", "Filters")}</button></div>
        {filtersOpen && <div className="filter-panel">
          <label>{tr("地域", "Location")}<select value={city} onChange={(event) => setCity(event.target.value)}><option value="">{tr("すべて", "All")}</option>{facets.cities.map((value) => <option key={`${value.ja}-${value.en}`} value={value.ja}>{localized(value, language)}</option>)}</select></label>
          <label>{tr("年", "Year")}<select value={year} onChange={(event) => setYear(event.target.value)}><option value="">{tr("すべて", "All")}</option>{facets.years.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label>{tr("形式", "Format")}<select value={format} onChange={(event) => setFormat(event.target.value)}><option value="">{tr("すべて", "All")}</option>{facets.formats.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>schema<select value={schemaStatus} onChange={(event) => setSchemaStatus(event.target.value)}><option value="">{tr("すべて", "All")}</option><option value="ready">{tr("あり", "Available")}</option><option value="missing">{tr("なし", "Missing")}</option><option value="invalid">{tr("エラー", "Error")}</option></select></label>
          <label>{tr("利用条件", "Terms of use")}<select value={license} onChange={(event) => setLicense(event.target.value)}><option value="">{tr("すべて", "All")}</option>{facets.licenses.map((value) => <option key={`${value.ja}-${value.en}`} value={value.ja}>{localized(value, language)}</option>)}</select></label>
          <button type="button" onClick={() => { setCity(""); setYear(""); setFormat(""); setSchemaStatus(""); setLicense(""); }}>{tr("解除", "Clear")}</button>
        </div>}
        <p className="result-summary"><strong>{filteredFiles.length}</strong> files / <strong>{visibleDatasets.length}</strong> datasets</p>
        {loadingError && <p className="error-message">{tr("カタログを読み込めませんでした．もう一度ページを開き直してください．", "The catalog could not be loaded. Please reload the page.")}</p>}
        {!loadingError && files.length > 0 && visibleDatasets.length === 0 && <p className="empty-message">{tr("条件に一致するデータがありません．検索語や絞り込みを変更してください．", "No data matches these conditions. Try changing the search term or filters.")}</p>}
        <div className="dataset-grid">
          {visibleDatasets.map((dataset, index) => {
            const matches = grouped.get(dataset.id) ?? [];
            const datasetFileIds = files.filter((file) => file.dataset_id === dataset.id).map((file) => file.id);
            const allSelected = datasetFileIds.length > 0 && datasetFileIds.every((id) => selected.has(id));
            const sample = localized(matches.find((file) => file.name)?.name ?? null, language);
            return <article className="dataset-card" key={dataset.id}>
              <div className="card-topline"><span className="index">{String(index + 1).padStart(2, "0")}</span><label className="check-label"><input type="checkbox" checked={allSelected} onChange={() => toggleDataset(dataset.id)}/><span>{tr("データセット全体", "Entire dataset")}</span></label></div>
              <p className="dataset-id">{dataset.id}</p><h3>{datasetLabel(dataset.id)}</h3>{sample && <p className="sample-name">{sample} {tr("など", "and more")}</p>}
              <p className="description">{language === "ja" ? `${matches.length}件が現在の条件に一致 / 全${dataset.file_count}ファイル` : `${matches.length} match the current filters / ${dataset.file_count} files total`}，{humanSize(dataset.total_size)}</p>
              <div className="tags"><span>{dataset.schema_ready_count} schemas</span>{[...new Set(matches.map((file) => file.format))].slice(0, 3).map((value) => <span key={value}>{value.toUpperCase()}</span>)}</div>
              <button className="detail-link" type="button" onClick={() => setActiveDataset(dataset.id)}>{tr("ファイルを見る", "View files")} <span aria-hidden="true">→</span></button>
            </article>;
          })}
        </div>
      </section>

      <section className="guide-section" id="guide">
        <p className="section-kicker">HOW TO USE</p><h2>{tr("見つけた後は，モデルへ．", "From discovery to modeling.")}</h2>
        <div className="guide-grid"><div><span>01</span><h3>{tr("schemaから探す", "Search the schemas")}</h3><p>{tr("名称だけでなく，地域，年，列名や説明から必要なファイルを探せます．", "Find files by location, year, column name, description, and more.")}</p></div><div><span>02</span><h3>{tr("必要なものを選ぶ", "Choose what you need")}</h3><p>{tr("データセット全体でも，ファイル単位でも選択できます．合計容量も確認できます．", "Select an entire dataset or individual files and check the total size.")}</p></div><div><span>03</span><h3>{tr("取得方法を選ぶ", "Choose a download method")}</h3><p>{tr("すぐに保存するならWeb，階層を保って研究環境へ配置するならCLIを選べます．", "Use the web for immediate downloads, or the CLI to place files in your research environment with their directory structure intact.")}</p></div></div>
        <div className="download-method-guide">
          <p className="method-kicker">WEB OR CLI?</p>
          <h3>{tr("二つの取得方法がある理由", "Why Davis offers two download methods")}</h3>
          <p className="method-intro">{tr("どちらも同じ実データを取得しますが，重視する点が異なります．Webは導入不要の手軽さを，CLIは階層構造と再現性を優先しています．", "Both methods retrieve the same data, but they serve different priorities: the web requires no installation, while the CLI prioritizes directory structure and reproducibility.")}</p>
          <div className="method-grid">
            <article className="method-card"><span className="method-badge">WEB DOWNLOAD</span><h4>{tr("少数のファイルを，すぐに確認する", "Quickly inspect a few files")}</h4><dl>
              <div><dt>{tr("向いている場面", "Best for")}</dt><dd>{tr("検索しながら数件を試したいとき，CLIを導入していないPCを使うとき，まずデータの中身を確認したいとき．", "Trying a few files while searching, using a computer without the CLI, or inspecting the data before setting up a project.")}</dd></div>
              <div><dt>{tr("メリット", "Advantages")}</dt><dd>{tr("インストールやターミナル操作が不要です．選択した実データ，schema.yaml，説明PDFをその場で保存できます．", "No installation or terminal is required. Save the selected data, schema.yaml files, and documentation PDFs directly from the catalog.")}</dd></div>
              <div><dt>{tr("注意点", "Trade-offs")}</dt><dd>{tr("各ファイルはブラウザのダウンロード先へ個別に保存され，data以下の階層は再現されません．同名ファイルはブラウザにより改名されることがあり，多数の同時ダウンロードが確認・制限される場合もあります．", "Files are saved individually to the browser's download location, without recreating the hierarchy below data. Browsers may rename duplicate filenames or ask for permission before allowing many downloads.")}</dd></div>
            </dl></article>
            <article className="method-card"><span className="method-badge">CLI DOWNLOAD</span><h4>{tr("研究用の配置を，そのまま再現する", "Recreate a research-ready layout")}</h4><dl>
              <div><dt>{tr("向いている場面", "Best for")}</dt><dd>{tr("データセット全体や多数のファイルを取得するとき，分析プロジェクトに配置するとき，同じ取得条件を後から再実行・共有したいとき．", "Downloading entire datasets or many files, placing them into an analysis project, or rerunning and sharing the same selection later.")}</dd></div>
              <div><dt>{tr("メリット", "Advantages")}</dt><dd>{tr("data/種類/データセット以下の階層を出力先に再現します．Webで選んだファイルとschema・PDFの指定がコピーしたコマンドへ反映されるため，取得条件を残せます．", "Recreates the hierarchy under data/type/dataset in the output directory. The copied command records the selected files and schema/PDF options, making the retrieval conditions reproducible.")}</dd></div>
              <div><dt>{tr("注意点", "Trade-offs")}</dt><dd>{tr("最初にDavis CLIの導入が必要です．保存したいディレクトリでターミナルを開いてコマンドを実行し，未ログインの場合は参加者用招待コードを入力します．", "The Davis CLI must be installed first. Open a terminal in the destination directory, run the command, and enter the participant invitation code if you are not already logged in.")}</dd></div>
            </dl><a href={language === "ja" ? "https://github.com/bin-utokyo/davis/blob/main/docs/participant-installation.md" : "https://github.com/bin-utokyo/davis/blob/main/docs/participant-installation_en.md"} target="_blank" rel="noreferrer">{tr("CLIの利用者向け導入ガイド", "CLI participant installation guide")} →</a></article>
          </div>
          <p className="method-recommendation"><strong>{tr("迷った場合", "If you are unsure")}</strong>{tr("数ファイルを試す段階ではWeb，実際の研究フォルダへ保存する段階ではCLIがおすすめです．Webで選択してから，そのままCLIコマンドをコピーできます．", "Start with the web when trying a few files, then use the CLI when saving them into a real research folder. You can make the selection on the web and copy the matching CLI command directly.")}</p>
        </div>
      </section>
      <footer id="about"><a className="brand" href="#top"><span className="brand-mark">D</span><span>Davis</span></a><p>{tr("交通データの取得から行動モデルの研究までを，一つの流れにつなぐためのプラットフォームです．", "A platform connecting transport data discovery and travel behavior model research in one workflow.")}</p><a href="https://github.com/bin-utokyo/davis">GitHub</a></footer>

      {activeDataset && <div className="overlay"><button className="overlay-dismiss" type="button" aria-label={tr("ファイル一覧を閉じる", "Close file list")} onClick={() => setActiveDataset(null)}/><aside className="drawer" role="dialog" aria-modal="true" aria-label={tr(`${activeDataset}のファイル`, `${activeDataset} files`)}>
        <div className="drawer-heading"><div><p className="dataset-id">{activeDataset}</p><h2>{datasetLabel(activeDataset)}</h2></div><button type="button" aria-label={tr("閉じる", "Close")} onClick={() => setActiveDataset(null)}>×</button></div>
        <p className="drawer-summary">{activeDatasetFiles.length} files ・ {humanSize(activeDatasetFiles.reduce((sum, file) => sum + file.size, 0))}</p>
        <div className="file-list">{activeDatasetFiles.map((file) => <div className="file-row" key={file.id}><input aria-label={tr(`${file.file_id}を選択`, `Select ${file.file_id}`)} type="checkbox" checked={selected.has(file.id)} onChange={() => toggleFile(file.id)}/><button type="button" onClick={() => setActiveFile(file)}><strong>{localized(file.name, language) || file.file_id}</strong><span>{file.file_id} ・ {humanSize(file.size)} ・ {file.schema_status === "ready" ? tr("schemaあり", "schema available") : tr("schemaなし", "schema unavailable")}</span></button></div>)}</div>
      </aside></div>}

      {activeFile && <div className="overlay detail-overlay"><button className="overlay-dismiss" type="button" aria-label={tr("ファイル詳細を閉じる", "Close file details")} onClick={() => setActiveFile(null)}/><aside className="drawer file-detail" role="dialog" aria-modal="true" aria-label={tr(`${activeFile.file_id}の詳細`, `Details for ${activeFile.file_id}`)}>
        <div className="drawer-heading"><div><p className="dataset-id">{activeFile.dataset_id}</p><h2>{localized(activeFile.name, language) || activeFile.file_id}</h2></div><button type="button" aria-label={tr("閉じる", "Close")} onClick={() => setActiveFile(null)}>×</button></div>
        {localized(activeFile.description, language) && <p className="file-description">{localized(activeFile.description, language)}</p>}
        <dl className="file-meta"><div><dt>{tr("ファイル", "File")}</dt><dd>{activeFile.file_id}</dd></div><div><dt>{tr("地域", "Location")}</dt><dd>{localized(activeFile.city, language) || tr("記載なし", "Not provided")}</dd></div><div><dt>{tr("年", "Year")}</dt><dd>{activeFile.year ?? tr("記載なし", "Not provided")}</dd></div><div><dt>{tr("形式・容量", "Format and size")}</dt><dd>{activeFile.format.toUpperCase()} ・ {humanSize(activeFile.size)}</dd></div></dl>
        {localized(activeFile.license, language) && <div className="license-note"><strong>{tr("利用条件", "Terms of use")}</strong><p>{localized(activeFile.license, language)}</p></div>}
        <h3 className="detail-title">{tr("列定義", "Column definitions")} ({activeFile.columns.length})</h3><div className="column-table">{activeFile.columns.map((column) => <div key={column.name}><code>{column.name}</code><span>{column.data_type}</span><p>{localized(column.description, language) || tr("説明なし", "No description")}</p></div>)}</div>
        {activeFile.raw_schema && <details className="raw-schema"><summary>Raw schema.yaml</summary><pre>{activeFile.raw_schema}</pre></details>}
      </aside></div>}

      {selected.size > 0 && <aside className="selection-dock" id="selection" aria-label={tr("選択内容", "Selection details")}><div><strong>{selected.size} files</strong><span>{humanSize(selectedSize)}</span></div><p>{tr(`${new Set(selectedFiles.map((file) => file.dataset_id)).size}データセットから選択中`, `Selected from ${new Set(selectedFiles.map((file) => file.dataset_id)).size} datasets`)}</p><button type="button" className="clear-button" onClick={() => setSelected(new Set())}>{tr("すべて解除", "Clear all")}</button><button type="button" className="copy-button" onClick={openDownloadDialog}>{tr("取得方法を選ぶ", "Choose download method")}</button></aside>}

      {downloadDialogOpen && <div className="overlay access-overlay"><button className="overlay-dismiss" type="button" aria-label={tr("ダウンロード画面を閉じる", "Close download dialog")} disabled={downloadPending} onClick={() => setDownloadDialogOpen(false)}/><section className="access-dialog" role="dialog" aria-modal="true" aria-labelledby="download-title">
        <div className="drawer-heading"><div><p className="dataset-id">{selectedFiles.length > 0 ? "DOWNLOAD" : "ACCESS"}</p><h2 id="download-title">{selectedFiles.length > 0 ? tr("選択内容を取得する", "Download your selection") : tr("参加者ログイン", "Participant login")}</h2></div><button type="button" aria-label={tr("閉じる", "Close")} disabled={downloadPending} onClick={() => setDownloadDialogOpen(false)}>×</button></div>
        {selectedFiles.length > 0 && <><div className="download-summary"><strong>{selectedDownloadCount} files</strong><span>{humanSize(selectedDownloadSize)}</span></div>
        <p className="download-note">{tr("Webでは各ファイルをブラウザのダウンロードフォルダへ保存します．", "On the web, each file is saved to your browser's Downloads folder.")}</p>
        <fieldset className="document-options"><legend>{tr("付属資料", "Companion documents")}</legend>
          <label className={!selectedHasSchema ? "document-option-unavailable" : undefined}><input type="checkbox" checked={selectedHasSchema && includeSchema} disabled={!selectedHasSchema} onChange={(event) => setIncludeSchema(event.target.checked)}/><span>schema.yaml</span></label>
          {selectedHasSchema && !includeSchema && <p className="document-warning">{tr("schema.yamlを含めない場合，列定義や利用条件が保存されず，将来Davisの整形・推定機能へ接続するときに再取得が必要になることがあります．", "Without schema.yaml, column definitions and terms of use are not saved locally, and you may need to download them again for future Davis formatting and modeling workflows.")}</p>}
          <label className={!selectedHasPdfJa ? "document-option-unavailable" : undefined}><input type="checkbox" checked={selectedHasPdfJa && includePdfJa} disabled={!selectedHasPdfJa} onChange={(event) => setIncludePdfJa(event.target.checked)}/><span>{tr("日本語説明PDF", "Japanese PDF")}</span></label>
          <label className={!selectedHasPdfEn ? "document-option-unavailable" : undefined}><input type="checkbox" checked={selectedHasPdfEn && includePdfEn} disabled={!selectedHasPdfEn} onChange={(event) => setIncludePdfEn(event.target.checked)}/><span>{tr("英語説明PDF", "English PDF")}</span></label>
          <p>{tr(`実データ${selectedFiles.length}件 + 付属資料${selectedSchemaDocuments.length + selectedPdfDocuments.length}件`, `${selectedFiles.length} data files + ${selectedSchemaDocuments.length + selectedPdfDocuments.length} companion documents`)}</p>
        </fieldset>
        <div className="cli-copy"><p>{tr("CLIならデータセットの階層構造を保って保存できます．上の付属資料の選択内容もコマンドに反映されます．", "The CLI preserves the dataset directory structure and applies the companion-document choices above.")} <a href={language === "ja" ? "https://github.com/bin-utokyo/davis/blob/main/docs/participant-installation.md" : "https://github.com/bin-utokyo/davis/blob/main/docs/participant-installation_en.md"} target="_blank" rel="noreferrer">{tr("利用者向け導入ガイド", "Participant installation guide")}</a></p><button type="button" onClick={copyCommands}>{copied ? tr("コピーしました", "Copied") : tr("CLIコマンドをコピー", "Copy CLI command")}</button></div>
        {selectedLicenses.length > 0 && <div className="license-list"><strong>{tr("利用条件", "Terms of use")}</strong>{selectedLicenses.map((value) => <p key={value}>{value}</p>)}</div>}
        </>}
        {sessionState !== "authenticated" && <form className="login-form" onSubmit={login}><label htmlFor="invite-code">{tr("参加者用招待コード", "Participant invitation code")}</label><div><input id="invite-code" type="password" autoComplete="off" required maxLength={256} value={inviteCode} onChange={(event) => setInviteCode(event.target.value)} placeholder={tr("運営から案内されたコード", "Code provided by the organizers")}/><button type="submit" disabled={authPending}>{authPending ? tr("確認中", "Checking") : tr("ログイン", "Log in")}</button></div><p>{tr("一度ログインすると，このブラウザではセッション期限まで再入力不要です．", "After logging in once, you will not need to enter the code again in this browser until the session expires.")}</p></form>}
        {sessionState === "authenticated" && <p className="session-status">{tr("ログイン済み", "Logged in")}{sessionExpiresAt && <> ・ {tr("セッション期限", "session expires")} {new Intl.DateTimeFormat(language === "ja" ? "ja-JP" : "en-US").format(new Date(sessionExpiresAt))}</>}</p>}
        {selectedFiles.length > 0 && <label className="license-confirm"><input type="checkbox" checked={licenseConfirmed} onChange={(event) => setLicenseConfirmed(event.target.checked)}/><span>{selectedLicenses.length > 0 ? tr("上記の利用条件と保存方法を確認しました．", "I have reviewed the terms of use and download method above.") : tr("保存方法を確認しました．", "I have reviewed the download method.")}</span></label>}
        {accessError && <p className="access-message error-message" role="alert">{accessError === "login" ? tr("ログインに失敗しました．招待コードを確認してください．", "Login failed. Please check the invitation code.") : accessError === "session" ? tr("セッションの有効期限が切れました．もう一度招待コードを入力してください．", "Your session has expired. Please enter the invitation code again.") : tr("ダウンロードを開始できませんでした．もう一度お試しください．", "The download could not be started. Please try again.")}</p>}
        {downloadCount > 0 && <p className="access-message success-message" role="status">{tr(`${downloadCount}ファイルのダウンロードを開始しました．`, `Started downloading ${downloadCount} files.`)}</p>}
        {selectedFiles.length > 0 && <button className="download-button" type="button" disabled={!licenseConfirmed || sessionState !== "authenticated" || downloadPending} onClick={downloadSelected}>{downloadPending ? tr("ダウンロードを準備中", "Preparing download") : tr(`${selectedDownloadCount}ファイルをブラウザでダウンロード`, `Download ${selectedDownloadCount} files in browser`)}</button>}
      </section></div>}
    </main>
  );
}
