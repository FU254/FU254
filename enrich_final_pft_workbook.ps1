param(
    [string]$DrWorkbook = "",
    [string]$SourceWorkbook = "",
    [string]$FinalWorkbook = "",
    [string]$BackupWorkbook = ""
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Resolve-WorkbookPath {
    param(
        [string]$ExplicitPath,
        [string]$SearchRoot,
        [string]$Pattern
    )
    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        if (-not (Test-Path -LiteralPath $ExplicitPath)) {
            throw "Workbook not found: $ExplicitPath"
        }
        return $ExplicitPath
    }
    $match = Get-ChildItem -Path $SearchRoot -Filter $Pattern | Select-Object -First 1
    if (-not $match) {
        throw "No workbook found under $SearchRoot with pattern $Pattern"
    }
    return $match.FullName
}

function Expand-WorkbookToTemp {
    param([string]$WorkbookPath)
    $tmpDir = Join-Path ([System.IO.Path]::GetTempPath()) ("xlsx_" + [guid]::NewGuid().ToString("N"))
    [System.IO.Directory]::CreateDirectory($tmpDir) | Out-Null
    $tmpWorkbook = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString("N") + ".xlsx")
    Copy-Item -LiteralPath $WorkbookPath -Destination $tmpWorkbook -Force
    [System.IO.Compression.ZipFile]::ExtractToDirectory($tmpWorkbook, $tmpDir)
    Remove-Item -LiteralPath $tmpWorkbook -Force
    return $tmpDir
}

function Get-SharedStrings {
    param([string]$ExtractedDir)
    $sharedPath = Join-Path $ExtractedDir "xl\sharedStrings.xml"
    $shared = @()
    if (-not (Test-Path -LiteralPath $sharedPath)) {
        return $shared
    }
    $raw = Get-Content -LiteralPath $sharedPath -Raw
    $matches = [regex]::Matches($raw, "<si\b.*?>.*?</si>", "Singleline")
    foreach ($match in $matches) {
        $siXml = $match.Value
        $textMatches = [regex]::Matches($siXml, "<t(?:\s+xml:space=""preserve"")?>(.*?)</t>", "Singleline")
        if ($textMatches.Count -eq 0) {
            $shared += ""
            continue
        }
        $parts = foreach ($textMatch in $textMatches) {
            [System.Net.WebUtility]::HtmlDecode($textMatch.Groups[1].Value)
        }
        $shared += ($parts -join "")
    }
    return $shared
}

function Get-SheetXmlPath {
    param(
        [string]$ExtractedDir,
        [string]$SheetName = "",
        [switch]$FirstSheet
    )
    [xml]$workbookXml = Get-Content -LiteralPath (Join-Path $ExtractedDir "xl\workbook.xml") -Raw
    [xml]$relsXml = Get-Content -LiteralPath (Join-Path $ExtractedDir "xl\_rels\workbook.xml.rels") -Raw
    $relMap = @{}
    foreach ($rel in $relsXml.Relationships.Relationship) {
        $relMap[$rel.Id] = $rel.Target
    }
    foreach ($sheet in $workbookXml.workbook.sheets.sheet) {
        if ($FirstSheet -or $sheet.name -eq $SheetName) {
            $rid = $sheet.GetAttribute("id", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
            if (-not $rid) { $rid = $sheet."r:id" }
            $target = $relMap[$rid]
            if (-not $target) { continue }
            if (-not $target.StartsWith("xl/")) {
                $target = "xl/" + $target.TrimStart("/")
            }
            return (Join-Path $ExtractedDir ($target -replace "/", "\"))
        }
    }
    throw "Could not locate sheet XML for sheet '$SheetName'"
}

function Get-ColumnLetters {
    param([string]$CellRef)
    return ($CellRef -replace "\d", "")
}

function ColumnLettersToIndex {
    param([string]$Letters)
    $result = 0
    foreach ($char in $Letters.ToCharArray()) {
        $result = ($result * 26) + ([int][char]$char - [int][char]'A' + 1)
    }
    return $result
}

function ColumnIndexToLetters {
    param([int]$Index)
    $letters = ""
    while ($Index -gt 0) {
        $Index--
        $letters = [char]([int][char]'A' + ($Index % 26)) + $letters
        $Index = [math]::Floor($Index / 26)
    }
    return $letters
}

function Get-CellText {
    param(
        $Cell,
        [string[]]$SharedStrings
    )
    if (-not $Cell) { return $null }
    $type = [string]$Cell.t
    if ($type -eq "s") {
        $idx = [int]$Cell.v
        if ($idx -lt $SharedStrings.Count) { return $SharedStrings[$idx] }
        return $null
    }
    if ($type -eq "inlineStr") {
        return [string]$Cell.is.t
    }
    return [string]$Cell.v
}

function Get-HeaderMap {
    param(
        $RowNode,
        [string[]]$SharedStrings
    )
    $map = @{}
    foreach ($cell in $RowNode.c) {
        $text = Get-CellText -Cell $cell -SharedStrings $SharedStrings
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            $map[$text] = Get-ColumnLetters ([string]$cell.r)
        }
    }
    return $map
}

function Get-CellByRef {
    param($RowNode, [string]$CellRef)
    foreach ($cell in $RowNode.c) {
        if ([string]$cell.r -eq $CellRef) {
            return $cell
        }
    }
    return $null
}

function Set-CellValue {
    param(
        [xml]$SheetXml,
        $RowNode,
        [string]$CellRef,
        $Value,
        [switch]$InlineString
    )
    $ns = $SheetXml.DocumentElement.NamespaceURI
    $cell = Get-CellByRef -RowNode $RowNode -CellRef $CellRef
    if (-not $cell) {
        $cell = $SheetXml.CreateElement("c", $ns)
        $cell.SetAttribute("r", $CellRef)
        $null = $RowNode.AppendChild($cell)
    }
    while ($cell.HasChildNodes) {
        $null = $cell.RemoveChild($cell.FirstChild)
    }
    $null = $cell.RemoveAttribute("t")

    if ($InlineString) {
        $cell.SetAttribute("t", "inlineStr")
        $is = $SheetXml.CreateElement("is", $ns)
        $t = $SheetXml.CreateElement("t", $ns)
        $t.InnerText = [string]$Value
        $null = $is.AppendChild($t)
        $null = $cell.AppendChild($is)
    }
    else {
        $v = $SheetXml.CreateElement("v", $ns)
        $v.InnerText = [string]$Value
        $null = $cell.AppendChild($v)
    }
}

function Save-ExtractedWorkbook {
    param(
        [string]$ExtractedDir,
        [string]$OutputPath
    )
    $tmpZip = $OutputPath + ".tmp"
    if (Test-Path -LiteralPath $tmpZip) {
        Remove-Item -LiteralPath $tmpZip -Force
    }
    if (Test-Path -LiteralPath $OutputPath) {
        Remove-Item -LiteralPath $OutputPath -Force
    }
    [System.IO.Compression.ZipFile]::CreateFromDirectory($ExtractedDir, $tmpZip)
    Move-Item -LiteralPath $tmpZip -Destination $OutputPath -Force
}

$root = "D:\99"
$DrWorkbook = Resolve-WorkbookPath -ExplicitPath $DrWorkbook -SearchRoot $root -Pattern "DR*2023.xlsx"
$SourceWorkbook = Resolve-WorkbookPath -ExplicitPath $SourceWorkbook -SearchRoot $root -Pattern "*2025.7.14-*.xlsx"
if ([string]::IsNullOrWhiteSpace($FinalWorkbook)) {
    $FinalWorkbook = (Get-ChildItem -Path $root -Filter "*.xlsx" |
        Where-Object { $_.Name -notlike "*before_*" -and $_.Length -eq 294300 } |
        Select-Object -First 1).FullName
}
if (-not $FinalWorkbook) {
    throw "Could not auto-detect final workbook."
}
if ([string]::IsNullOrWhiteSpace($BackupWorkbook)) {
    $BackupWorkbook = Join-Path $root "肺功能万能表最终版.before_weight_smoking.xlsx"
}

Write-Output "DR workbook: $DrWorkbook"
Write-Output "Source workbook: $SourceWorkbook"
Write-Output "Final workbook: $FinalWorkbook"
Write-Output "Backup workbook: $BackupWorkbook"

Copy-Item -LiteralPath $FinalWorkbook -Destination $BackupWorkbook -Force

$drDir = $null
$srcDir = $null
$finalDir = $null

try {
    $drDir = Expand-WorkbookToTemp -WorkbookPath $DrWorkbook
    $drShared = Get-SharedStrings -ExtractedDir $drDir
    $drSheetPath = Get-SheetXmlPath -ExtractedDir $drDir -FirstSheet
    [xml]$drSheet = Get-Content -LiteralPath $drSheetPath -Raw
    $drRows = $drSheet.worksheet.sheetData.row
    $drHeader = Get-HeaderMap -RowNode ($drRows | Where-Object { [int]$_.r -eq 1 }) -SharedStrings $drShared
    $drAccCol = $drHeader["放射编号"]
    $drSeqCol = $drHeader["序号"]
    $accessionToSeq = @{}
    foreach ($row in $drRows | Where-Object { [int]$_.r -ge 2 }) {
        $rowNum = [int]$row.r
        $acc = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $drAccCol, $rowNum)) -SharedStrings $drShared
        $seq = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $drSeqCol, $rowNum)) -SharedStrings $drShared
        if ([string]::IsNullOrWhiteSpace($acc) -or [string]::IsNullOrWhiteSpace($seq)) { continue }
        if (-not $accessionToSeq.ContainsKey($acc)) {
            $accessionToSeq[$acc] = $seq
        }
    }

    $srcDir = Expand-WorkbookToTemp -WorkbookPath $SourceWorkbook
    $srcShared = Get-SharedStrings -ExtractedDir $srcDir
    $srcSheetPath = Get-SheetXmlPath -ExtractedDir $srcDir -SheetName "临床数据"
    [xml]$srcSheet = Get-Content -LiteralPath $srcSheetPath -Raw
    $srcRows = $srcSheet.worksheet.sheetData.row
    $srcHeader = Get-HeaderMap -RowNode ($srcRows | Where-Object { [int]$_.r -eq 2 }) -SharedStrings $srcShared
    $srcSeqCol = $srcHeader["序号"]
    $srcWeightCol = $srcHeader["体重"]
    $srcSmokingCol = $srcHeader["吸烟"]
    $seqToWeight = @{}
    $seqToSmoking = @{}
    foreach ($row in $srcRows | Where-Object { [int]$_.r -ge 3 }) {
        $rowNum = [int]$row.r
        $seq = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $srcSeqCol, $rowNum)) -SharedStrings $srcShared
        if ([string]::IsNullOrWhiteSpace($seq)) { continue }
        if (-not $seqToWeight.ContainsKey($seq)) {
            $seqToWeight[$seq] = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $srcWeightCol, $rowNum)) -SharedStrings $srcShared
        }
        if (-not $seqToSmoking.ContainsKey($seq)) {
            $seqToSmoking[$seq] = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $srcSmokingCol, $rowNum)) -SharedStrings $srcShared
        }
    }

    $finalDir = Expand-WorkbookToTemp -WorkbookPath $FinalWorkbook
    $finalShared = Get-SharedStrings -ExtractedDir $finalDir
    $finalSheetPath = Get-SheetXmlPath -ExtractedDir $finalDir -FirstSheet
    [xml]$finalSheet = Get-Content -LiteralPath $finalSheetPath -Raw
    $finalRows = $finalSheet.worksheet.sheetData.row
    $headerRow = $finalRows | Where-Object { [int]$_.r -eq 1 }
    $finalHeader = Get-HeaderMap -RowNode $headerRow -SharedStrings $finalShared
    $finalAccCol = $finalHeader["放射编号"]
    if (-not $finalAccCol) { throw "Final workbook is missing 放射编号 column." }

    $existingCols = @()
    foreach ($cell in $headerRow.c) {
        $existingCols += ColumnLettersToIndex (Get-ColumnLetters ([string]$cell.r))
    }
    $lastColIndex = ($existingCols | Measure-Object -Maximum).Maximum
    $weightColLetters = if ($finalHeader.ContainsKey("weight")) { $finalHeader["weight"] } else { ColumnIndexToLetters ($lastColIndex + 1) }
    $smokingColLetters = if ($finalHeader.ContainsKey("smoking")) { $finalHeader["smoking"] } else { ColumnIndexToLetters ($lastColIndex + 2) }

    Set-CellValue -SheetXml $finalSheet -RowNode $headerRow -CellRef ($weightColLetters + "1") -Value "weight" -InlineString
    Set-CellValue -SheetXml $finalSheet -RowNode $headerRow -CellRef ($smokingColLetters + "1") -Value "smoking" -InlineString

    $updatedWeight = 0
    $updatedSmoking = 0
    $missingAccession = 0
    $missingSeq = 0

    foreach ($row in $finalRows | Where-Object { [int]$_.r -ge 2 }) {
        $rowNum = [int]$row.r
        $acc = Get-CellText -Cell (Get-CellByRef -RowNode $row -CellRef ("{0}{1}" -f $finalAccCol, $rowNum)) -SharedStrings $finalShared
        if ([string]::IsNullOrWhiteSpace($acc)) {
            $missingAccession++
            continue
        }
        if (-not $accessionToSeq.ContainsKey($acc)) {
            $missingSeq++
            continue
        }
        $seq = $accessionToSeq[$acc]
        if ($seqToWeight.ContainsKey($seq) -and -not [string]::IsNullOrWhiteSpace([string]$seqToWeight[$seq])) {
            Set-CellValue -SheetXml $finalSheet -RowNode $row -CellRef ($weightColLetters + $rowNum) -Value $seqToWeight[$seq]
            $updatedWeight++
        }
        if ($seqToSmoking.ContainsKey($seq) -and -not [string]::IsNullOrWhiteSpace([string]$seqToSmoking[$seq])) {
            Set-CellValue -SheetXml $finalSheet -RowNode $row -CellRef ($smokingColLetters + $rowNum) -Value $seqToSmoking[$seq]
            $updatedSmoking++
        }
    }

    $maxRow = ($finalRows | Measure-Object -Property r -Maximum).Maximum
    $finalSheet.worksheet.dimension.ref = ("A1:{0}{1}" -f $smokingColLetters, $maxRow)
    $finalSheet.Save($finalSheetPath)
    Save-ExtractedWorkbook -ExtractedDir $finalDir -OutputPath $FinalWorkbook

    Write-Output "Updated workbook: $FinalWorkbook"
    Write-Output "Rows with weight written: $updatedWeight"
    Write-Output "Rows with smoking written: $updatedSmoking"
    Write-Output "Rows missing accession: $missingAccession"
    Write-Output "Rows missing seq mapping: $missingSeq"
    Write-Output "Added/used columns: weight=$weightColLetters, smoking=$smokingColLetters"
}
finally {
    foreach ($dir in @($drDir, $srcDir, $finalDir)) {
        if ($dir -and (Test-Path -LiteralPath $dir)) {
            try { Remove-Item -LiteralPath $dir -Recurse -Force } catch {}
        }
    }
}
