<#
Sailfish :: harness — agentic tool-run rig. Drives Ollama OR the Sailfish vLLM container,
so you benchmark the same tool-run tests against either backend on the same card.

  pwsh harness/tool_harness.ps1                                   # Ollama (gemma4:e4b @ 11434)
  pwsh harness/tool_harness.ps1 -Backend openai -Port 22343       # Sailfish container (OpenAI API)
  pwsh harness/tool_harness.ps1 -Show                             # print every tool call + result

Reports per case: right-tool? valid-args? turns, tokens, decode TPS, latency — plus a summary
with tool-call accuracy and avg TPS. That TPS is the number Sailfish has to push past Ollama.
#>
param(
  [ValidateSet("ollama","openai")] [string]$Backend = "ollama",
  [string]$Model = "gemma4:e4b",
  [string]$Url,
  [int]$Port,
  [int]$MaxTurns = 6,
  [switch]$Show
)
$ErrorActionPreference = "Stop"
if (-not $Port) { $Port = if ($Backend -eq "ollama") { 11434 } else { 22343 } }
if (-not $Url)  { $Url  = if ($Backend -eq "ollama") { "http://localhost:$Port" } else { "http://localhost:$Port/v1" } }

# ---- tools advertised to the model (same schema works for Ollama + OpenAI) ----
$ToolSchemas = @(
  @{ type="function"; function=@{ name="calculator"; description="Evaluate an arithmetic expression and return the number.";
      parameters=@{ type="object"; properties=@{ expression=@{ type="string"; description="e.g. (47*89)+12" } }; required=@("expression") } } },
  @{ type="function"; function=@{ name="get_current_time"; description="Return the current local date and time.";
      parameters=@{ type="object"; properties=@{}; required=@() } } },
  @{ type="function"; function=@{ name="list_files"; description="List file names in a directory.";
      parameters=@{ type="object"; properties=@{ path=@{ type="string"; description="Directory path" } }; required=@("path") } } },
  @{ type="function"; function=@{ name="get_weather"; description="Get the current weather for a city (demo stub).";
      parameters=@{ type="object"; properties=@{ city=@{ type="string"; description="City name" } }; required=@("city") } } }
)

function Invoke-Tool([string]$Name, $ToolArgs) {
  switch ($Name) {
    "calculator" {
      $safe = ([string]$ToolArgs.expression) -replace '[^0-9\.\+\-\*\/\(\)\s]', ''
      if (-not $safe) { return "ERROR: empty/unsafe expression" }
      try { return [string]([double](Invoke-Expression $safe)) } catch { return "ERROR: $($_.Exception.Message)" }
    }
    "get_current_time" { return (Get-Date -Format "yyyy-MM-dd HH:mm:ss K") }
    "list_files" {
      $p = [string]$ToolArgs.path
      if (-not (Test-Path $p)) { return "ERROR: path not found: $p" }
      return (Get-ChildItem -Path $p -Name -ErrorAction SilentlyContinue | Select-Object -First 40) -join ", "
    }
    "get_weather" { return "22C, sunny, wind 8 km/h (stub for $([string]$ToolArgs.city))" }
    default { return "ERROR: unknown tool '$Name'" }
  }
}

# ---- one chat round, normalized across backends ----
function Invoke-Chat($Messages) {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  if ($Backend -eq "ollama") {
    $body = @{ model=$Model; messages=$Messages; tools=$ToolSchemas; stream=$false } | ConvertTo-Json -Depth 20
    $r = Invoke-RestMethod -Uri "$Url/api/chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    $sw.Stop()
    $raw = $r.message
    $tcs = @(); foreach ($tc in @($r.message.tool_calls)) { if ($tc) { $tcs += [pscustomobject]@{ id=$tc.id; name=$tc.function.name; args=$tc.function.arguments } } }
    $tps = if ($r.eval_count -and $r.eval_duration) { [math]::Round($r.eval_count / ($r.eval_duration/1e9), 1) } else { 0 }
    return [pscustomobject]@{ Raw=$raw; ToolCalls=$tcs; Content=[string]$r.message.content; TPS=$tps; LatencyMs=[math]::Round($sw.Elapsed.TotalMilliseconds,0) }
  } else {
    $body = @{ model=$Model; messages=$Messages; tools=$ToolSchemas; stream=$false } | ConvertTo-Json -Depth 20
    $r = Invoke-RestMethod -Uri "$Url/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 300
    $sw.Stop()
    $raw = $r.choices[0].message
    $tcs = @(); foreach ($tc in @($raw.tool_calls)) { if ($tc) {
      $a = $tc.function.arguments; if ($a -is [string]) { try { $a = $a | ConvertFrom-Json } catch {} }
      $tcs += [pscustomobject]@{ id=$tc.id; name=$tc.function.name; args=$a } } }
    $ct = if ($r.usage.completion_tokens) { [int]$r.usage.completion_tokens } else { 0 }
    # llama.cpp returns true decode speed in timings.predicted_per_second; wall-clock (incl. prefill) is the fallback
    $tps = if ($r.timings -and $r.timings.predicted_per_second) { [math]::Round($r.timings.predicted_per_second, 1) }
           elseif ($ct -gt 0 -and $sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($ct / $sw.Elapsed.TotalSeconds, 1) } else { 0 }
    return [pscustomobject]@{ Raw=$raw; ToolCalls=$tcs; Content=[string]$raw.content; TPS=$tps; LatencyMs=[math]::Round($sw.Elapsed.TotalMilliseconds,0) }
  }
}

function Run-Case($Prompt, $Expect) {
  $messages = [System.Collections.ArrayList]@()
  [void]$messages.Add(@{ role="user"; content=$Prompt })
  $turns=0; $tpsList=@(); $latTotal=0; $firstTool=$null; $toolsUsed=@(); $final=""; $anyErr=$false
  for ($turns=1; $turns -le $MaxTurns; $turns++) {
    $resp = Invoke-Chat $messages
    $latTotal += $resp.LatencyMs; if ($resp.TPS -gt 0 -and $resp.TPS -lt 1000) { $tpsList += $resp.TPS }
    if ($resp.ToolCalls.Count -gt 0) {
      [void]$messages.Add($resp.Raw)
      foreach ($tc in $resp.ToolCalls) {
        if (-not $firstTool) { $firstTool = $tc.name }
        $toolsUsed += $tc.name
        $result = Invoke-Tool $tc.name $tc.args
        if ([string]$result -like "ERROR*") { $anyErr=$true }
        if ($Show) { Write-Host "  [tool] $($tc.name)($($tc.args | ConvertTo-Json -Compress)) -> $result" -ForegroundColor DarkCyan }
        if ($Backend -eq "ollama") { [void]$messages.Add(@{ role="tool"; tool_name=$tc.name; content=[string]$result }) }
        else { [void]$messages.Add(@{ role="tool"; tool_call_id=$tc.id; content=[string]$result }) }
      }
      continue
    }
    $final = $resp.Content
    if ($Show) { Write-Host "  [final] $final" -ForegroundColor DarkGray }
    break
  }
  $pass = if ($null -eq $Expect) { ($firstTool -eq $null) } else { ($firstTool -eq $Expect) }
  $avgTps = if ($tpsList.Count) { [math]::Round(($tpsList | Measure-Object -Average).Average,1) } else { 0 }
  [pscustomobject]@{ Prompt=$Prompt; Expect=($Expect ?? "(chat)"); GotTool=($firstTool ?? "(none)"); Pass=$pass; ToolErr=$anyErr; Turns=$turns; AvgTPS=$avgTps; LatencyMs=$latTotal }
}

$Cases = @(
  @{ Prompt="What is (47*89)+12? Use a tool."; Expect="calculator" },
  @{ Prompt="What time is it right now?"; Expect="get_current_time" },
  @{ Prompt="List the files in C:\Windows."; Expect="list_files" },
  @{ Prompt="What's the weather in Tokyo?"; Expect="get_weather" },
  @{ Prompt="Compute 15% of 200, then tell me the current time."; Expect="calculator" },
  @{ Prompt="Who wrote the novel 1984? Answer directly, no tools."; Expect=$null }
)

Write-Host "== Sailfish harness ==  backend=$Backend  model=$Model  url=$Url" -ForegroundColor Cyan
$results = @()
foreach ($c in $Cases) {
  Write-Host "`n> $($c.Prompt)" -ForegroundColor White
  try { $r = Run-Case $c.Prompt $c.Expect } catch { Write-Host "  ERR: $($_.Exception.Message)" -ForegroundColor Red; continue }
  $tag = if ($r.Pass -and -not $r.ToolErr) { "PASS" } elseif ($r.Pass) { "PASS*" } else { "FAIL" }
  $col = if ($r.Pass -and -not $r.ToolErr) { "Green" } elseif ($r.Pass) { "Yellow" } else { "Red" }
  Write-Host ("  {0}  tool={1}  turns={2}  tps={3}  {4}ms{5}" -f $tag,$r.GotTool,$r.Turns,$r.AvgTPS,$r.LatencyMs,$(if($r.ToolErr){"  (tool error!)"}else{""})) -ForegroundColor $col
  $results += $r
}
$passN = ($results | Where-Object Pass).Count
$tpsAll = if ($results.Count) { [math]::Round(($results | Where-Object AvgTPS -gt 0 | Measure-Object AvgTPS -Average).Average,1) } else { 0 }
Write-Host "`n== Summary ($Backend) ==" -ForegroundColor Cyan
$results | Format-Table Prompt,Expect,GotTool,Pass,ToolErr,Turns,AvgTPS,LatencyMs -AutoSize
Write-Host ("tool-call accuracy: {0}/{1}   avg decode TPS: {2}   backend: {3}" -f $passN,$results.Count,$tpsAll,$Backend) -ForegroundColor Cyan
