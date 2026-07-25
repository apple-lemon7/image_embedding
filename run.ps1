$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $projectDir 'bin\Release\net10.0-windows\ImageEmbedding.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error 'The application is not built. Run setup.ps1 first.'
    exit 1
}
Start-Process -FilePath $exe -WorkingDirectory (Split-Path -Parent $exe)
