# rebuild_docx.ps1
# ================
# Regenerates BOTH .docx copies from the corrected paper Markdown.
#
# Why two copies: paper/ holds the submission artefact, and docs/assets/ holds
# the copy served by the GitHub Pages site. They must stay identical or the
# public site shows stale numbers. Write them from the same source in one step.
#
# Usage (from anywhere):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\rebuild_docx.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\rebuild_docx.ps1 -Converter "tools\md2docx.js"
#
# Requires: Node.js, and a Markdown-to-DOCX converter (or Pandoc).

param(
    # Node executable. Override if `node` is not on PATH.
    [string]$Node = "node",

    # Path to the md2docx.js converter script.
    [string]$Converter = "tools\md2docx.js",

    # Document title used for the Word title page and running header.
    [string]$Title = "Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models"
)

$ErrorActionPreference = "Stop"

# Resolve the repo root from this script's location, so the script is portable
# and does not depend on where it is invoked from.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$src  = Join-Path $repo "paper\CSE432_Bangla_Sentiment_BERT_Paper.md"
$dst1 = Join-Path $repo "paper\CSE432_Bangla_Sentiment_BERT_Paper.docx"
$dst2 = Join-Path $repo "docs\assets\CSE432_Bangla_Sentiment_BERT_Paper.docx"

# Fail loudly if source is missing; fall back to pandoc if md2docx path not found.
if (-not (Test-Path $src)) { throw "source Markdown not found: $src" }

$usePandoc = $false
if (-not (Test-Path $Converter)) {
    if (Get-Command pandoc -ErrorAction SilentlyContinue) {
        Write-Output "md2docx converter not found at '$Converter'; falling back to system pandoc."
        $usePandoc = $true
    } else {
        throw "Neither md2docx converter ($Converter) nor pandoc was found.`nPass -Converter <path> to scripts\md2docx.js or install pandoc."
    }
}

Write-Output "repo      : $repo"
Write-Output "source    : $src"
if ($usePandoc) {
    Write-Output "engine    : pandoc"
} else {
    Write-Output "converter : $Converter"
}
Write-Output ""

foreach ($dst in @($dst1, $dst2)) {
    if ($usePandoc) {
        & pandoc $src --resource-path="paper;." -o $dst
    } else {
        & $Node $Converter $src $dst --theme=standard --title=$Title
    }
    if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE) writing $dst" }
    Write-Output "[write] $dst"
}

Write-Output ""
Write-Output "--- sizes ---"
foreach ($f in @($dst1, $dst2)) {
    $i = Get-Item $f
    Write-Output ("{0,10:N0} bytes  {1}" -f $i.Length, $f)
}

Write-Output ""
Write-Output "Next: run tools\verify_docx.ps1 to confirm no fabricated values remain."
