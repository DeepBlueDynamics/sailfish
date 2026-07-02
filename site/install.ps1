# Sailfish local installer (Windows / PowerShell).
#   irm https://sailfish.nuts.services/install.ps1 | iex
$ErrorActionPreference = "Stop"
Write-Host "sailfish - sovereign fast inference" -ForegroundColor Cyan

# 1. docker present?
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Write-Host "Docker Desktop not found. Install it (with the NVIDIA/WSL2 GPU integration) then re-run." -ForegroundColor Yellow
  Write-Host "  https://www.docker.com/products/docker-desktop/"; return
}

# 2. GPU passthrough check
Write-Host "checking GPU passthrough into containers..." -ForegroundColor DarkCyan
$gpu = docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
if ($LASTEXITCODE -ne 0 -or -not $gpu) {
  Write-Host "No GPU visible to Docker. Enable WSL2 GPU support in Docker Desktop settings, then re-run." -ForegroundColor Yellow
  Write-Host "CPU-only fallback works but is slow.";
}
else { Write-Host "GPU: $gpu" -ForegroundColor Green }

# 3. pull + run the appliance (llama.cpp tier B by default; autodetect promotes >=16GB to vLLM)
$vram = 0
if ($gpu) { try { $vram = [double](($gpu -split ",")[1] -replace "[^0-9.]","") / 1024 } catch {} }
Write-Host "pulling deepbluedynamics/sailfish ..." -ForegroundColor DarkCyan
docker pull deepbluedynamics/sailfish:latest
docker rm -f sailfish 2>$null | Out-Null

# Mount Claude Code transcripts read-only so the appliance can harvest tool calls locally.
# Read-only, never uploaded — the training data stays on your box.
$claudeArgs = @()
$claudeDir = Join-Path $env:USERPROFILE ".claude"
if (Test-Path $claudeDir) {
  $claudeArgs = @("-v", "${claudeDir}:/root/.claude:ro")
  Write-Host "mounting transcripts (read-only): $claudeDir" -ForegroundColor DarkCyan
}
docker run -d --name sailfish --gpus all -p 22343:22343 `
  -v sailfish-cache:/root/.cache `
  @claudeArgs `
  --restart unless-stopped deepbluedynamics/sailfish:latest
Write-Host ""
Write-Host "sailfish is up -> http://localhost:22343" -ForegroundColor Green
Write-Host "OpenAI-compatible API at http://localhost:22343/v1"
Start-Process "http://localhost:22343"
