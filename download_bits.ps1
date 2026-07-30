$dest = "F:\Amit\BCI_Classification\C-\Users\Amit_Addi\mne_data\MNE-bnci-data\~bci\database\001-2014"
$base = "https://lampx.tugraz.at/~bci/database/001-2014/"
$MIN_SIZE = 30MB

$subjects = 1..9
$sessions = @('T', 'E')

$jobs = @()

foreach ($s in $subjects) {
    foreach ($m in $sessions) {
        $fname = "A0{0}{1}.mat" -f $s, $m
        $target = Join-Path $dest $fname
        
        # Skip if already fully downloaded
        if ((Test-Path $target) -and (Get-Item $target).Length -ge $MIN_SIZE) {
            Write-Host "[SKIP] $fname already downloaded ($(([math]::Round((Get-Item $target).Length / 1MB, 1))) MB)"
            continue
        }
        # Remove partial file
        if (Test-Path $target) { Remove-Item $target -Force }
        
        $url = $base + $fname
        Write-Host "[QUEUE] $fname -> $target"
        $jobs += @{Source = $url; Destination = $target; Name = $fname}
    }
}

Write-Host "`nStarting $($jobs.Count) BITS transfers..."
$bitsJobs = @()
foreach ($j in $jobs) {
    $bj = Start-BitsTransfer -Source $j.Source -Destination $j.Destination `
        -DisplayName $j.Name -Priority Foreground -Asynchronous
    $bitsJobs += $bj
    Write-Host "[STARTED] $($j.Name) JobId=$($bj.JobId)"
}

Write-Host "`nMonitoring downloads..."
$done = 0
while ($done -lt $bitsJobs.Count) {
    Start-Sleep 5
    $done = 0
    foreach ($bj in $bitsJobs) {
        $bj = Get-BitsTransfer -JobId $bj.JobId -ErrorAction SilentlyContinue
        if ($bj -eq $null) { $done++; continue }
        if ($bj.JobState -eq 'Transferred') {
            Complete-BitsTransfer $bj
            Write-Host "[DONE] $($bj.DisplayName)"
            $done++
        } elseif ($bj.JobState -eq 'Error') {
            Write-Host "[ERROR] $($bj.DisplayName): $($bj.ErrorDescription)"
            $done++
        } else {
            $pct = if ($bj.BytesTotal -gt 0) { [math]::Round($bj.BytesTransferred / $bj.BytesTotal * 100, 1) } else { 0 }
            Write-Host "[PROGRESS] $($bj.DisplayName): $pct% ($($bj.BytesTransferred) / $($bj.BytesTotal))"
        }
    }
}
Write-Host "`nAll downloads complete!"
