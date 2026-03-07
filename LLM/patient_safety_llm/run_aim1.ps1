param(
    [ValidateSet('auto', 'grok', 'openai', 'ollama', 'llama_cpp')]
    [string]$Provider = 'auto',

    [string]$Data = 'LLM/temp_scenarios.csv',

    [string]$Model,

    [string]$BaseUrl,

    [string]$LlamaServerUrl,

    [string]$ApiKey
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$scriptPath = Join-Path $scriptDir "run_real_llm_study.py"

if (-not (Test-Path $pythonExe)) {
    throw "Python environment not found: $pythonExe"
}

if (-not (Test-Path $scriptPath)) {
    throw "Study runner not found: $scriptPath"
}

$cliArgs = @($scriptPath, '--provider', $Provider, '--data', $Data)

if ($Model) {
    $cliArgs += @('--model', $Model)
}
if ($BaseUrl) {
    $cliArgs += @('--base-url', $BaseUrl)
}
if ($LlamaServerUrl) {
    $cliArgs += @('--llama-server-url', $LlamaServerUrl)
}
if ($ApiKey) {
    $cliArgs += @('--api-key', $ApiKey)
}

Write-Host "Running Aim1 with provider=$Provider data=$Data"
Push-Location $repoRoot
try {
    & $pythonExe @cliArgs
}
finally {
    Pop-Location
}
