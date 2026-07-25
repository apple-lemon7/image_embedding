$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectDir '.venv\Scripts\python.exe'

Write-Host 'Creating the project-local Python environment...'
if (-not (Test-Path -LiteralPath $venvPython)) {
    python -m venv (Join-Path $projectDir '.venv')
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the Python environment (exit code $LASTEXITCODE)." }
}

Write-Host 'Installing ONNX Runtime, Transformers, FAISS and dependencies...'
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw "Failed to initialize pip (exit code $LASTEXITCODE)." }
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip (exit code $LASTEXITCODE)." }
& $venvPython -m pip install -r (Join-Path $projectDir 'worker\requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "Failed to install Python dependencies (exit code $LASTEXITCODE)." }

Write-Host 'Building the .NET WPF application in Release mode...'
dotnet build (Join-Path $projectDir 'ImageEmbedding.csproj') -c Release
if ($LASTEXITCODE -ne 0) { throw "Failed to build the WPF application (exit code $LASTEXITCODE)." }

Write-Host 'Setup complete. Start bin\Release\net10.0-windows\ImageEmbedding.exe.'
