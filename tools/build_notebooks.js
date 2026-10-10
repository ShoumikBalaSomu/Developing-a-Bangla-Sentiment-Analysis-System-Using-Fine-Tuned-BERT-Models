#!/usr/bin/env node
/**
 * build_notebooks.js
 * ==================
 * Assembles the three Colab notebooks from human-editable cell source files.
 *
 * Why this exists: .ipynb is strict JSON with a fixed nbformat schema. Editing
 * it by hand is error-prone, and a malformed notebook silently fails to open in
 * Colab. Keeping the cells as plain text and generating the JSON means the
 * notebooks are reviewable in a diff and always schema-valid.
 *
 * Cell source format:
 *     ###CELL md###        <- starts a markdown cell
 *     ###CELL code###      <- starts a code cell
 *     ...cell body...
 *
 * Usage:
 *     node tools/build_notebooks.js
 *     node tools/build_notebooks.js --check    # validate without writing
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const SRC_DIR = path.join(ROOT, "tools", "notebooks");
const OUT_DIR = path.join(ROOT, "notebooks");

const DELIM = /^###CELL (md|code)###\s*$/;

// Notebook-level metadata. `accelerator` and `colab.name` are what make the
// "Open in Colab" badge land on a GPU runtime with a sensible filename.
const NOTEBOOK_SPECS = [
  {
    file: "01_data_preprocessing",
    accelerator: "CPU",
    gpuType: null,
  },
  {
    file: "02_bert_finetuning",
    accelerator: "GPU",
    gpuType: "T4",
  },
  {
    file: "03_evaluation_and_analysis",
    accelerator: "GPU",
    gpuType: "T4",
  },
];

/**
 * Split a cell-source file into [{type, body}] preserving order.
 * Body lines keep their trailing newline, which is what nbformat expects:
 * every source line except the last carries "\n".
 */
function parseCells(text, srcName) {
  const lines = text.split(/\r?\n/);
  const cells = [];
  let cur = null;

  for (const line of lines) {
    const m = line.match(DELIM);
    if (m) {
      if (cur) cells.push(cur);
      cur = { type: m[1] === "md" ? "markdown" : "code", lines: [] };
      continue;
    }
    if (cur) cur.lines.push(line);
  }
  if (cur) cells.push(cur);

  if (cells.length === 0) {
    throw new Error(`${srcName}: no ###CELL### markers found`);
  }

  return cells.map((c) => {
    // Trim a single leading and trailing blank line for readability, but keep
    // intentional internal blank lines.
    while (c.lines.length && c.lines[0].trim() === "") c.lines.shift();
    while (c.lines.length && c.lines[c.lines.length - 1].trim() === "") c.lines.pop();
    return { type: c.type, body: c.lines };
  });
}

/** Convert body lines into the nbformat `source` array form. */
function toSource(lines) {
  if (lines.length === 0) return [];
  return lines.map((l, i) => (i === lines.length - 1 ? l : l + "\n"));
}

function buildNotebook(spec, cells) {
  const nbCells = cells.map((c) => {
    if (c.type === "markdown") {
      return {
        cell_type: "markdown",
        metadata: {},
        source: toSource(c.body),
      };
    }
    return {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: toSource(c.body),
    };
  });

  const colabMeta = {
    name: `${spec.file}.ipynb`,
    provenance: [],
    toc_visible: true,
    collapsed_sections: [],
  };
  if (spec.accelerator === "GPU") {
    colabMeta.accelerator = "GPU";
    if (spec.gpuType) colabMeta.gpuType = spec.gpuType;
  }

  return {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      colab: colabMeta,
      kernelspec: { display_name: "Python 3", name: "python3" },
      language_info: { name: "python", version: "3.10" },
    },
    cells: nbCells,
  };
}

/** Cheap lint: catch the mistakes that break a notebook at runtime. */
function lint(spec, cells) {
  const problems = [];
  cells.forEach((c, i) => {
    const body = c.body.join("\n");
    if (c.type === "code") {
      if (body.includes("\t")) {
        problems.push(`cell ${i}: tab character in code (use spaces)`);
      }
      // A markdown fence left inside a code cell means a mis-split source file.
      if (/^#{1,6}\s/m.test(body) && !body.includes("#@title") && !body.includes("#@markdown")) {
        problems.push(`cell ${i}: looks like markdown inside a code cell`);
      }
      if (body.includes("###CELL")) {
        problems.push(`cell ${i}: nested ###CELL marker`);
      }
    }
  });
  return problems;
}

function main() {
  const checkOnly = process.argv.includes("--check");
  fs.mkdirSync(OUT_DIR, { recursive: true });

  let failed = 0;
  for (const spec of NOTEBOOK_SPECS) {
    const srcPath = path.join(SRC_DIR, `${spec.file}.cells.md`);
    const outPath = path.join(OUT_DIR, `${spec.file}.ipynb`);

    if (!fs.existsSync(srcPath)) {
      console.error(`[skip] ${srcPath} not found`);
      failed++;
      continue;
    }

    const cells = parseCells(fs.readFileSync(srcPath, "utf8"), spec.file);
    const problems = lint(spec, cells);
    if (problems.length) {
      console.error(`[lint] ${spec.file}:`);
      problems.forEach((p) => console.error(`   - ${p}`));
      failed++;
      continue;
    }

    const nb = buildNotebook(spec, cells);
    const json = JSON.stringify(nb, null, 1);

    // Round-trip parse proves the file Colab will receive is valid JSON.
    const reparsed = JSON.parse(json);
    const nMd = reparsed.cells.filter((c) => c.cell_type === "markdown").length;
    const nCode = reparsed.cells.filter((c) => c.cell_type === "code").length;

    if (checkOnly) {
      console.log(`[ok]   ${spec.file}: ${nMd} md + ${nCode} code cells (validated, not written)`);
      continue;
    }

    fs.writeFileSync(outPath, json, "utf8");
    console.log(`[write] ${path.relative(ROOT, outPath)}  (${nMd} md + ${nCode} code cells, ${(json.length / 1024).toFixed(1)} KB)`);
  }

  if (failed) {
    console.error(`\n${failed} notebook(s) failed`);
    process.exit(1);
  }
  console.log(checkOnly ? "\nall notebooks valid" : "\ndone");
}

main();
