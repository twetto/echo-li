#requires -Version 5.1
# EqF landmark-cap sweep over the whole EuRoC set.
# Varies ONLY eqf.maxFeatures; tracker pool (RudolfV.maxFeatures) and sparse
# depth bank (SparseVog.max_pool_size) stay at 300. Isolates "do more features
# in the EqF state improve the trajectory?" No --vis (batch, unattended).
$ErrorActionPreference = 'Continue'
$root = 'C:\Users\twetto\Documents\devs\echo-li'
Set-Location $root

$featCounts = @(40, 100, 150)
$baseCfg = 'configs\eqvio_euroc_rho.yaml'
$outRoot = 'sweep_results'
New-Item -ItemType Directory -Force -Path "$outRoot\configs" | Out-Null

# name + dataset path (mirrors run.ps1; the two V2_02 paths get distinct names).
$seqs = @(
  @{ name = 'V1_01_easy';      path = '..\python-vio\V1_01_easy' },
  @{ name = 'V2_02_medium_pv'; path = '..\python-vio\V2_02_medium' },
  @{ name = 'V1_03_difficult'; path = '..\..\..\Downloads\vicon_room1\vicon_room1\V1_03_difficult' },
  @{ name = 'V2_01_easy';      path = '..\..\..\Downloads\vicon_room2\vicon_room2\V2_01_easy\V2_01_easy' },
  @{ name = 'V2_02_medium';    path = '..\..\..\Downloads\vicon_room2\vicon_room2\V2_02_medium\V2_02_medium' },
  @{ name = 'V2_03_difficult'; path = '..\..\..\Downloads\vicon_room2\vicon_room2\V2_03_difficult\V2_03_difficult' },
  @{ name = 'MH_01_easy';      path = '..\..\..\Downloads\machine_hall\machine_hall\MH_01_easy\MH_01_easy' },
  @{ name = 'MH_02_easy';      path = '..\..\..\Downloads\machine_hall\machine_hall\MH_02_easy\MH_02_easy' },
  @{ name = 'MH_03_medium';    path = '..\..\..\Downloads\machine_hall\machine_hall\MH_03_medium\MH_03_medium' },
  @{ name = 'MH_04_difficult'; path = '..\..\..\Downloads\machine_hall\machine_hall\MH_04_difficult\MH_04_difficult' },
  @{ name = 'MH_05_difficult'; path = '..\..\..\Downloads\machine_hall\machine_hall\MH_05_difficult\MH_05_difficult' }
)

"[build] cargo build -p echo-li-cli --release --features rerun"
cargo build -p echo-li-cli --release --features rerun
if ($LASTEXITCODE -ne 0) { 'BUILD FAILED'; exit 1 }
$bin = 'target\release\echo-li-cli.exe'

foreach ($fc in $featCounts) {
  # Patch ONLY the eqf block's "maxFeatures: 40" line (RudolfV's 300 and the
  # commented variants are left untouched). Write BOM-free (serde_yaml chokes on BOM).
  $cfg = "$outRoot\configs\eqvio_mf$fc.yaml"
  (Get-Content $baseCfg) -replace '^\s*maxFeatures:\s*40\b.*$', "  maxFeatures: $fc           # core EqF landmark cap (sweep)" |
    Set-Content -Encoding ascii $cfg
  foreach ($s in $seqs) {
    $out = "$outRoot\mf$fc\$($s.name)"
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    "[run] mf=$fc $($s.name)"
    & $bin -d $s.path -c $cfg --output $out *> "$out\run.log"
    "      exit=$LASTEXITCODE"
  }
}

# Summarize ATE (position RMSE + attitude RMSE) into a table + CSV.
$rows = foreach ($fc in $featCounts) {
  foreach ($s in $seqs) {
    $mf = "$outRoot\mf$fc\$($s.name)\trajectory_metrics.txt"
    if (Test-Path $mf) {
      $c = Get-Content $mf
      $pos = ($c | Select-String '^ate_position_rmse_m (.+)$').Matches.Groups[1].Value
      $att = ($c | Select-String '^ate_attitude_rmse_deg (.+)$').Matches.Groups[1].Value
      [pscustomobject]@{ maxFeat = $fc; seq = $s.name; pos_rmse_m = [double]$pos; att_rmse_deg = [double]$att }
    }
    else {
      [pscustomobject]@{ maxFeat = $fc; seq = $s.name; pos_rmse_m = 'FAIL'; att_rmse_deg = 'FAIL' }
    }
  }
}
$rows | Export-Csv -NoTypeInformation -Encoding ascii "$outRoot\summary.csv"
$rows | Format-Table -AutoSize | Out-String
'SWEEP DONE'
