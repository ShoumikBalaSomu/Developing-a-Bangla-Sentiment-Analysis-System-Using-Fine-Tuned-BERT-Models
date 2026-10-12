# Verify the fabricated ablation value is gone from the .docx binaries,
# and that the corrected value is present.
#
# A .docx is a zip; the body text lives in word/document.xml. Grep cannot read
# it, so unzip in memory and search the XML directly.

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

# Resolve the repo root from this script's location so the check is portable.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Bad strings that must NOT appear anywhere (fabricated / overclaimed).
$bad = @("0.5768", "+2.17", "-2.17", "2.17%")
# Good strings that MUST appear (the real measurement and the honest framing).
$good = @("0.6228", "2.43", "negative delta")

$targets = @(
    "paper\CSE432_Bangla_Sentiment_BERT_Paper.docx",
    "docs\assets\CSE432_Bangla_Sentiment_BERT_Paper.docx"
)

$fail = $false

foreach ($f in $targets) {
    if (-not (Test-Path $f)) { Write-Output "MISSING FILE: $f"; $fail = $true; continue }

    $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $f))
    try {
        $entry = $zip.Entries | Where-Object { $_.FullName -eq "word/document.xml" }
        $reader = New-Object System.IO.StreamReader($entry.Open())
        $xml = $reader.ReadToEnd()
        $reader.Close()
    } finally {
        $zip.Dispose()
    }

    # Strip XML tags so wrapped runs do not split a number across elements.
    $text = [regex]::Replace($xml, "<[^>]+>", "")

    Write-Output "=== $f ==="
    Write-Output ("body text length: {0:N0} chars" -f $text.Length)

    foreach ($b in $bad) {
        if ($text.Contains($b)) {
            Write-Output "  [FAIL] fabricated value still present: $b"
            $fail = $true
        } else {
            Write-Output "  [ok] absent: $b"
        }
    }
    foreach ($g in $good) {
        if ($text.Contains($g)) {
            Write-Output "  [ok] present: $g"
        } else {
            Write-Output "  [FAIL] corrected value missing: $g"
            $fail = $true
        }
    }
    Write-Output ""
}

if ($fail) { Write-Output "RESULT: VERIFICATION FAILED"; exit 1 }
Write-Output "RESULT: ALL CHECKS PASSED"
exit 0
