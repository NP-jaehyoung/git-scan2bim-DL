param(
    [string]$PyRevitExtensionsDir = "$env:APPDATA\\pyRevit\\Extensions"
)

$source = Join-Path $PSScriptRoot "pyrevit\\Scan2BIM.extension"
$target = Join-Path $PyRevitExtensionsDir "Scan2BIM.extension"

if (-not (Test-Path $source)) {
    throw "Source extension folder not found: $source"
}

New-Item -ItemType Directory -Force $PyRevitExtensionsDir | Out-Null
if (Test-Path $target) {
    Remove-Item $target -Recurse -Force
}
Copy-Item $source $target -Recurse -Force

Write-Host "Installed pyRevit extension to $target"
