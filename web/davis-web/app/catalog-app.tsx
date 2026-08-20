"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type LocalizedText = { ja: string; en: string };
type Column = { name: string; data_type: string; description: LocalizedText | null };
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

export function CatalogApp() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [files, setFiles] = useState<CatalogFile[]>([]);
  const [facets, setFacets] = useState<Facets>(emptyFacets);
  const [loadingError, setLoadingError] = useState("");
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

  useEffect(() => {
    Promise.all([
      fetch("/catalog/datasets.json").then((response) => response.json() as Promise<Dataset[]>),
      fetch("/catalog/files.json").then((response) => response.json() as Promise<CatalogFile[]>),
      fetch("/catalog/facets.json").then((response) => response.json() as Promise<Facets>),
    ]).then(([nextDatasets, nextFiles, nextFacets]) => {
      setDatasets(nextDatasets);
      setFiles(nextFiles);
      setFacets(nextFacets);
    }).catch(() => setLoadingError("カタログを読み込めませんでした．もう一度ページを開き直してください．"));
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
    const ids = [...new Set(selectedFiles.map((file) => file.dataset_id))];
    const commands = ids.map((id) => {
      const datasetFiles = files.filter((file) => file.dataset_id === id);
      const chosenFiles = selectedFiles.filter((file) => file.dataset_id === id);
      if (datasetFiles.length === chosenFiles.length) return `davis get ${id}`;
      return `davis get ${id} ${chosenFiles.map((file) => `--file ${JSON.stringify(file.file_id)}`).join(" ")}`;
    }).join("\n");
    await navigator.clipboard.writeText(commands);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Davis catalog home"><span className="brand-mark">D</span><span>Davis</span></a>
        <nav aria-label="メインナビゲーション"><a className="active" href="#catalog">データを探す</a><a href="#guide">使い方</a><a href="#about">Davisについて</a></nav>
        <button className="selection-button" type="button" onClick={() => selected.size && document.querySelector("#selection")?.scrollIntoView()}>
          選択中 <span>{selected.size}</span>
        </button>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow">TRANSPORT DATA CATALOG</div>
        <h1>交通データを，<br />研究のすぐそばに．</h1>
        <p>データの内容と列定義を確かめながら，必要なファイルをまとめて探せる交通行動研究のためのカタログです．</p>
        <form className="search-panel" role="search" onSubmit={submitSearch}>
          <label htmlFor="catalog-search">データを検索</label>
          <div className="search-row"><span aria-hidden="true">⌕</span><input id="catalog-search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="地域，年，列名，データの説明から検索"/><button type="submit">検索</button></div>
          <div className="quick-filters" aria-label="検索例"><span>検索例</span>{["松山", "2021", "travel_time"].map((value) => <button key={value} type="button" onClick={() => applyQuickSearch(value)}>{value}</button>)}</div>
        </form>
        <div className="hero-stats" aria-label="カタログ概要"><div><strong>{datasets.length || "–"}</strong><span>datasets</span></div><div><strong>{files.length || "–"}</strong><span>files</span></div><div><strong>{files.filter((file) => file.schema_status === "ready").length || "–"}</strong><span>schemas</span></div></div>
      </section>

      <section className="catalog-section" id="catalog">
        <div className="section-heading"><div><p className="section-kicker">CATALOG</p><h2>データセットを探す</h2></div><button className="filter-button" type="button" aria-expanded={filtersOpen} onClick={() => setFiltersOpen((value) => !value)}>絞り込み</button></div>
        {filtersOpen && <div className="filter-panel">
          <label>地域<select value={city} onChange={(event) => setCity(event.target.value)}><option value="">すべて</option>{facets.cities.map((value) => <option key={`${value.ja}-${value.en}`} value={value.ja}>{value.ja}</option>)}</select></label>
          <label>年<select value={year} onChange={(event) => setYear(event.target.value)}><option value="">すべて</option>{facets.years.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label>形式<select value={format} onChange={(event) => setFormat(event.target.value)}><option value="">すべて</option>{facets.formats.map((value) => <option key={value}>{value}</option>)}</select></label>
          <label>schema<select value={schemaStatus} onChange={(event) => setSchemaStatus(event.target.value)}><option value="">すべて</option><option value="ready">あり</option><option value="missing">なし</option><option value="invalid">エラー</option></select></label>
          <label>license<select value={license} onChange={(event) => setLicense(event.target.value)}><option value="">すべて</option>{facets.licenses.map((value) => <option key={`${value.ja}-${value.en}`} value={value.ja}>{value.ja}</option>)}</select></label>
          <button type="button" onClick={() => { setCity(""); setYear(""); setFormat(""); setSchemaStatus(""); setLicense(""); }}>解除</button>
        </div>}
        <p className="result-summary"><strong>{filteredFiles.length}</strong> files / <strong>{visibleDatasets.length}</strong> datasets</p>
        {loadingError && <p className="error-message">{loadingError}</p>}
        {!loadingError && files.length > 0 && visibleDatasets.length === 0 && <p className="empty-message">条件に一致するデータがありません．検索語や絞り込みを変更してください．</p>}
        <div className="dataset-grid">
          {visibleDatasets.map((dataset, index) => {
            const matches = grouped.get(dataset.id) ?? [];
            const datasetFileIds = files.filter((file) => file.dataset_id === dataset.id).map((file) => file.id);
            const allSelected = datasetFileIds.length > 0 && datasetFileIds.every((id) => selected.has(id));
            const sample = matches.find((file) => file.name)?.name?.ja;
            return <article className="dataset-card" key={dataset.id}>
              <div className="card-topline"><span className="index">{String(index + 1).padStart(2, "0")}</span><label className="check-label"><input type="checkbox" checked={allSelected} onChange={() => toggleDataset(dataset.id)}/><span>dataset全体</span></label></div>
              <p className="dataset-id">{dataset.id}</p><h3>{datasetLabel(dataset.id)}</h3>{sample && <p className="sample-name">{sample} など</p>}
              <p className="description">{matches.length}件が現在の条件に一致 / 全{dataset.file_count} files，{humanSize(dataset.total_size)}</p>
              <div className="tags"><span>{dataset.schema_ready_count} schemas</span>{[...new Set(matches.map((file) => file.format))].slice(0, 3).map((value) => <span key={value}>{value.toUpperCase()}</span>)}</div>
              <button className="detail-link" type="button" onClick={() => setActiveDataset(dataset.id)}>ファイルを見る <span aria-hidden="true">→</span></button>
            </article>;
          })}
        </div>
      </section>

      <section className="guide-section" id="guide"><p className="section-kicker">HOW TO USE</p><h2>見つけた後は，モデルへ．</h2><div className="guide-grid"><div><span>01</span><h3>schemaから探す</h3><p>名称だけでなく，地域，年，列名や説明から必要なファイルを探せます．</p></div><div><span>02</span><h3>必要なものを選ぶ</h3><p>dataset全体でも，ファイル単位でも選択できます．合計容量も確認できます．</p></div><div><span>03</span><h3>Davisで取得する</h3><p>現在はCLI commandをコピーします．R2接続後はWebからの取得も追加します．</p></div></div></section>
      <footer id="about"><a className="brand" href="#top"><span className="brand-mark">D</span><span>Davis</span></a><p>交通データの取得から行動モデルの研究までを，一つの流れにつなぐためのplatformです．</p><a href="https://github.com/bin-utokyo/davis">GitHub</a></footer>

      {activeDataset && <div className="overlay"><button className="overlay-dismiss" type="button" aria-label="ファイル一覧を閉じる" onClick={() => setActiveDataset(null)}/><aside className="drawer" role="dialog" aria-modal="true" aria-label={`${activeDataset}のファイル`}>
        <div className="drawer-heading"><div><p className="dataset-id">{activeDataset}</p><h2>{datasetLabel(activeDataset)}</h2></div><button type="button" aria-label="閉じる" onClick={() => setActiveDataset(null)}>×</button></div>
        <p className="drawer-summary">{activeDatasetFiles.length} files ・ {humanSize(activeDatasetFiles.reduce((sum, file) => sum + file.size, 0))}</p>
        <div className="file-list">{activeDatasetFiles.map((file) => <div className="file-row" key={file.id}><input aria-label={`${file.file_id}を選択`} type="checkbox" checked={selected.has(file.id)} onChange={() => toggleFile(file.id)}/><button type="button" onClick={() => setActiveFile(file)}><strong>{file.name?.ja ?? file.file_id}</strong><span>{file.file_id} ・ {humanSize(file.size)} ・ {file.schema_status === "ready" ? "schemaあり" : "schemaなし"}</span></button></div>)}</div>
      </aside></div>}

      {activeFile && <div className="overlay detail-overlay"><button className="overlay-dismiss" type="button" aria-label="ファイル詳細を閉じる" onClick={() => setActiveFile(null)}/><aside className="drawer file-detail" role="dialog" aria-modal="true" aria-label={`${activeFile.file_id}の詳細`}>
        <div className="drawer-heading"><div><p className="dataset-id">{activeFile.dataset_id}</p><h2>{activeFile.name?.ja ?? activeFile.file_id}</h2></div><button type="button" aria-label="閉じる" onClick={() => setActiveFile(null)}>×</button></div>
        {activeFile.description?.ja && <p className="file-description">{activeFile.description.ja}</p>}
        <dl className="file-meta"><div><dt>ファイル</dt><dd>{activeFile.file_id}</dd></div><div><dt>地域</dt><dd>{activeFile.city?.ja ?? "記載なし"}</dd></div><div><dt>年</dt><dd>{activeFile.year ?? "記載なし"}</dd></div><div><dt>形式・容量</dt><dd>{activeFile.format.toUpperCase()} ・ {humanSize(activeFile.size)}</dd></div></dl>
        {activeFile.license?.ja && <div className="license-note"><strong>利用条件</strong><p>{activeFile.license.ja}</p></div>}
        <h3 className="detail-title">列定義 ({activeFile.columns.length})</h3><div className="column-table">{activeFile.columns.map((column) => <div key={column.name}><code>{column.name}</code><span>{column.data_type}</span><p>{column.description?.ja ?? "説明なし"}</p></div>)}</div>
        {activeFile.raw_schema && <details className="raw-schema"><summary>Raw schema.yaml</summary><pre>{activeFile.raw_schema}</pre></details>}
      </aside></div>}

      {selected.size > 0 && <aside className="selection-dock" id="selection" aria-label="選択内容"><div><strong>{selected.size} files</strong><span>{humanSize(selectedSize)}</span></div><p>{new Set(selectedFiles.map((file) => file.dataset_id)).size} datasetsから選択中</p><button type="button" className="clear-button" onClick={() => setSelected(new Set())}>すべて解除</button><button type="button" className="copy-button" onClick={copyCommands}>{copied ? "コピーしました" : "選択内容のgetをコピー"}</button></aside>}
    </main>
  );
}
