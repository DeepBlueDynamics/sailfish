<#
Sailfish :: profile — decode-speed profiler across output regimes, same prompts on either backend.

  pwsh harness/profile.ps1 -Backend openai -Port 22343 -Model gemma4-e4b     # Sailfish (llama.cpp+ngram)
  pwsh harness/profile.ps1 -Backend ollama -Model gemma4:e4b                  # Ollama baseline

Why regimes: n-gram speculation (Sailfish Tier B) drafts repeated tokens for free, so its win is
regime-dependent — big on repetitive/agentic output, ~neutral on prose. A single number hides that.
Reports true DECODE tok/s (llama.cpp timings.predicted_per_second / Ollama eval_count/eval_duration),
median of N reps, per regime. Greedy (temp 0) so both stacks decode the same tokens.
#>
param(
  [ValidateSet("ollama","openai")] [string]$Backend = "openai",
  [string]$Model,
  [int]$Port,
  [string]$Url,
  [int]$MaxTokens = 512,
  [int]$Reps = 3,
  [string]$Json   # optional: write results JSON here
)
$ErrorActionPreference = "Stop"
if (-not $Model) { $Model = if ($Backend -eq "ollama") { "gemma4:e4b" } else { "gemma4-e4b" } }
if (-not $Port)  { $Port  = if ($Backend -eq "ollama") { 11434 } else { 22343 } }
if (-not $Url)   { $Url   = if ($Backend -eq "ollama") { "http://localhost:$Port" } else { "http://localhost:$Port/v1" } }

# Same prompts both backends. Each is written to elicit a long output in one regime.
$Tasks = @(
  @{ name="prose";       prompt="Write a flowing ~400-word short story about a lighthouse keeper who finds a message in a bottle. Prose only, no lists or headings." },
  @{ name="code";        prompt="Write a complete, heavily commented Python implementation of a binary search tree: a Node class and a BST class with insert, search, and inorder traversal, plus a short usage example." },
  @{ name="repetitive";  prompt="List every integer from 1 to 80. Output each on its own line, formatted EXACTLY as: Row <n>: value=<n>, squared=<n*n>. Do not stop early; output all 80 lines." },
  @{ name="agentic-json";prompt="Output the following JSON object 25 times, one per line, incrementing only the id (1..25) and leaving everything else identical: {""id"":1,""tool"":""read_file"",""args"":{""path"":""/etc/hosts""},""status"":""ok""}" }
)

function Measure-Decode($Prompt) {
  $messages = @(@{ role="user"; content=$Prompt })
  $sw = [Diagnostics.Stopwatch]::StartNew()
  if ($Backend -eq "ollama") {
    $body = @{ model=$Model; messages=$messages; stream=$false; options=@{ temperature=0; num_predict=$MaxTokens } } | ConvertTo-Json -Depth 20
    $r = Invoke-RestMethod -Uri "$Url/api/chat" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    $sw.Stop()
    $toks = [int]$r.eval_count
    $tps  = if ($r.eval_count -and $r.eval_duration) { [math]::Round($r.eval_count / ($r.eval_duration/1e9), 1) } else { 0 }
  } else {
    $body = @{ model=$Model; messages=$messages; stream=$false; temperature=0; max_tokens=$MaxTokens } | ConvertTo-Json -Depth 20
    $r = Invoke-RestMethod -Uri "$Url/chat/completions" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    $sw.Stop()
    $toks = if ($r.usage.completion_tokens) { [int]$r.usage.completion_tokens } else { 0 }
    $tps  = if ($r.timings -and $r.timings.predicted_per_second) { [math]::Round($r.timings.predicted_per_second,1) }
            elseif ($toks -gt 0) { [math]::Round($toks / $sw.Elapsed.TotalSeconds,1) } else { 0 }
  }
  return [pscustomobject]@{ Tokens=$toks; TPS=$tps; WallMs=[math]::Round($sw.Elapsed.TotalMilliseconds,0) }
}

Write-Host "== Sailfish profiler ==  backend=$Backend  model=$Model  url=$Url  reps=$Reps  max_tokens=$MaxTokens" -ForegroundColor Cyan
$rows = @()
foreach ($t in $Tasks) {
  Write-Host "`n> [$($t.name)]" -ForegroundColor White
  $tpsList=@(); $tokList=@()
  for ($i=1; $i -le $Reps; $i++) {
    try { $m = Measure-Decode $t.prompt } catch { Write-Host "  ERR: $($_.Exception.Message)" -ForegroundColor Red; continue }
    $tpsList += $m.TPS; $tokList += $m.Tokens
    Write-Host ("  rep{0}: {1} tok  {2} tok/s  {3}ms" -f $i,$m.Tokens,$m.TPS,$m.WallMs) -ForegroundColor DarkGray
  }
  if ($tpsList.Count) {
    $sorted = $tpsList | Sort-Object
    $median = $sorted[[math]::Floor($sorted.Count/2)]
    $rows += [pscustomobject]@{ Regime=$t.name; MedianTPS=$median; MaxTPS=($tpsList|Measure-Object -Maximum).Maximum; AvgTokens=[math]::Round(($tokList|Measure-Object -Average).Average,0) }
  }
}
Write-Host "`n== Profile ($Backend / $Model) ==" -ForegroundColor Cyan
$rows | Format-Table Regime,MedianTPS,MaxTPS,AvgTokens -AutoSize
$overall = if ($rows.Count) { [math]::Round(($rows|Measure-Object MedianTPS -Average).Average,1) } else { 0 }
Write-Host ("overall median decode: {0} tok/s across {1} regimes" -f $overall,$rows.Count) -ForegroundColor Cyan
if ($Json) { @{ backend=$Backend; model=$Model; rows=$rows; overall=$overall } | ConvertTo-Json -Depth 6 | Set-Content $Json; Write-Host "wrote $Json" }
