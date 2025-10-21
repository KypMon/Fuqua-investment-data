const NORMALIZED_LINE_BREAK = /\r\n|\n|\r/g;

export const ensureCsvExtension = (filename) => {
  if (filename === null || filename === undefined) {
    return null;
  }

  const trimmed = String(filename).trim();
  if (trimmed.length === 0) {
    return null;
  }

  return trimmed.toLowerCase().endsWith(".csv") ? trimmed : `${trimmed}.csv`;
};

const escapeCsvValue = (value) => {
  const stringValue = value == null ? "" : String(value);
  if (stringValue === "") {
    return "";
  }

  const cleaned = stringValue.replace(NORMALIZED_LINE_BREAK, " ").trim();
  const needsEscaping = /[",\n]/.test(cleaned);

  if (!needsEscaping) {
    return cleaned;
  }

  return `"${cleaned.replace(/"/g, '""')}"`;
};

export const downloadCsvContent = (csvString, filename = "table-data.csv") => {
  const normalizedName = ensureCsvExtension(filename) ?? "table-data.csv";
  const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", normalizedName);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

const getCellText = (cell) => {
  if (!cell) {
    return "";
  }

  return cell.textContent.replace(/\s+/g, " ").trim();
};

export const extractTablesFromHtml = (htmlString) => {
  if (typeof window === "undefined" || typeof DOMParser === "undefined") {
    return [];
  }

  if (!htmlString) {
    return [];
  }

  try {
    const parser = new DOMParser();
    const doc = parser.parseFromString(htmlString, "text/html");
    const tables = Array.from(doc.querySelectorAll("table"));

    return tables
      .map((table, index) => {
        const caption = table.querySelector("caption")?.textContent?.trim() || null;
        const rows = Array.from(table.rows).map((row) =>
          Array.from(row.cells).map((cell) => getCellText(cell)),
        );

        return {
          id: index,
          caption,
          rows: rows.filter((row) => row.some((value) => value && value.length > 0)),
        };
      })
      .filter((table) => table.rows.length > 0);
  } catch (error) {
    console.error("Failed to parse HTML tables", error);
    return [];
  }
};

export const tablesToCsv = (tables) => {
  if (!Array.isArray(tables) || tables.length === 0) {
    return "";
  }

  const segments = tables.map((table, index) => {
    const lines = table.rows.map((row) => row.map(escapeCsvValue).join(","));
    const caption = table.caption ? `# ${table.caption}` : `# Table ${index + 1}`;
    return [caption, ...lines].join("\n");
  });

  return segments.join("\n\n");
};
