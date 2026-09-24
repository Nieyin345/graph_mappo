param(
    [switch]$NoQueue,
    [ValidateSet('pairhist', 'control')][string]$Family = 'pairhist'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$resultRoot = Join-Path $projectRoot $(if ($Family -eq 'control') { 'server_results\dminj_control' } else { 'server_results\pairhist_v2' })
$remoteHost = 'qinglong@clnode263.clemson.cloudlab.us'
$remoteRoot = '/opt/qkd/ppo-stock-fix-20260922'
$threadId = '01a0c8cf-4c68-7561-a2d7-3a182d7806b0'
$statusFile = Join-Path $resultRoot 'status.json'
$signalFile = Join-Path $resultRoot 'last_signaled.json'
$pollLog = Join-Path $resultRoot 'poll.log'
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

function Write-PollLog([string]$message) {
    Add-Content -LiteralPath $pollLog -Value "$(Get-Date -Format o) $message" -Encoding UTF8
}

function Copy-RemoteFile([string]$remoteFile, [string]$localFile) {
    & scp.exe -q -o BatchMode=yes -o ConnectTimeout=15 "${remoteHost}:$remoteFile" $localFile
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $remoteFile" }
}

try {
    $rawStatus = & ssh.exe -o BatchMode=yes -o ConnectTimeout=15 $remoteHost "/opt/qkd/venv/bin/python $remoteRoot/scripts/automation/pairhist_status.py --family $Family"
    if ($LASTEXITCODE -ne 0) { throw "ssh status failed: exit=$LASTEXITCODE" }
    $status = ($rawStatus -join "`n") | ConvertFrom-Json
    $status | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $statusFile -Encoding UTF8

    $previous = @{}
    if (Test-Path -LiteralPath $signalFile) {
        $stored = Get-Content -LiteralPath $signalFile -Raw | ConvertFrom-Json
        foreach ($property in $stored.PSObject.Properties) { $previous[$property.Name] = $property.Value }
    }
    $events = New-Object System.Collections.Generic.List[string]
    foreach ($run in $status.runs) {
        $localRun = Join-Path $resultRoot $run.run
        New-Item -ItemType Directory -Force -Path $localRun | Out-Null
        $remoteRun = "$remoteRoot/outputs/$($run.run)"
        Copy-RemoteFile "$remoteRun/resolved_config.yaml" (Join-Path $localRun 'resolved_config.yaml')
        Copy-RemoteFile "$remoteRun/train.log" (Join-Path $localRun 'train.log')
        if ($run.latest_update -gt 30) {
            Copy-RemoteFile "$remoteRun/metrics.jsonl" (Join-Path $localRun 'metrics.jsonl')
        }
        if ($run.latest_checkpoint) {
            $localCheckpoint = Join-Path $localRun $run.latest_checkpoint
            if (-not (Test-Path -LiteralPath $localCheckpoint)) {
                Copy-RemoteFile "$remoteRun/$($run.latest_checkpoint)" $localCheckpoint
            }
        }
        if ($run.complete) {
            $localFinal = Join-Path $localRun 'checkpoint_final.pt'
            if (-not (Test-Path -LiteralPath $localFinal)) {
                Copy-RemoteFile "$remoteRun/checkpoint_final.pt" $localFinal
            }
        }

        $checkpointUpdate = 0
        if ($run.latest_checkpoint -match 'checkpoint_update_(\d+)\.pt') {
            $checkpointUpdate = [int]$Matches[1]
        }
        $eventKey = if ($run.complete) {
            'complete'
        } elseif ($run.failed) {
            "failed:$($run.latest_update)"
        } elseif ($run.stalled) {
            "stalled:$($run.latest_update)"
        } elseif ($checkpointUpdate -gt 0 -and $run.validation_update -eq $checkpointUpdate) {
            "validation:$($run.validation_update)"
        } else {
            ''
        }
        if ($eventKey -and $previous[$run.run] -ne $eventKey) {
            $eventUpdate = if ($run.complete -and $null -ne $run.final_update) { $run.final_update } else { $run.latest_update }
            $events.Add("$($run.run): $eventKey, update=$eventUpdate")
            $previous[$run.run] = $eventKey
        }
    }

    $report = @(
        $(if ($Family -eq 'control') { '# QKD dminj 等预算控制组训练监控' } else { '# QKD pair-history v2 训练监控' })
        ''
        "更新时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz')"
        ''
    )
    if ($Family -eq 'control') {
        $report += '| Seed | 状态 | Update | 训练成功率 | KL | 裁剪率 | Critic相关性 | 验证成功率 | 相对旧30 | 相对v2 |'
        $report += '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
    } else {
        $report += '| Seed | 状态 | Update | 训练成功率 | KL | 裁剪率 | Critic相关性 | 验证成功率 | 对照差值 |'
        $report += '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'
    }
    foreach ($run in $status.runs) {
        $state = if ($run.complete) { '完成' } elseif ($run.failed) { '失败' } elseif ($run.stalled) { '停滞' } else { '运行中' }
        $m = $run.latest_metrics
        $v = $run.validation_comparison
        $displayUpdate = if ($run.complete -and $null -ne $run.final_update) { $run.final_update } else { $run.latest_update }
        $row = @(
            $run.seed, $state, $displayUpdate,
            $(if ($null -ne $m.mean_success_rate) { '{0:P2}' -f $m.mean_success_rate } else { '—' }),
            $(if ($null -ne $m.kl) { '{0:F4}' -f $m.kl } else { '—' }),
            $(if ($null -ne $m.clip_frac) { '{0:P1}' -f $m.clip_frac } else { '—' }),
            $(if ($null -ne $m.value_return_corr) { '{0:F3}' -f $m.value_return_corr } else { '—' }),
            $(if ($null -ne $v.current_mean_success_rate) { '{0:P2}' -f $v.current_mean_success_rate } else { '待评估' }),
            $(if ($null -ne $v.paired_mean_delta) { '{0:+0.00%;-0.00%;0.00%}' -f $v.paired_mean_delta } else { '—' })
        )
        if ($Family -eq 'control') {
            $p = $run.pairhist_comparison
            $row += $(if ($null -ne $p.paired_mean_delta_control_minus_pairhist) {
                '{0:+0.00%;-0.00%;0.00%}' -f $p.paired_mean_delta_control_minus_pairhist
            } else { '待评估' })
        }
        $report += '| ' + ($row -join ' | ') + ' |'
    }
    $report += @('', $(if ($Family -eq 'control') {
        '差值均为相同验证种子上的配对成功率差；“相对v2”是控制组减去 pair-history v2。'
    } else {
        '“对照差值”是相同验证种子上，相对于旧 dminj 模型的平均成功率差。'
    }))
    $report -join "`n" | Set-Content -LiteralPath (Join-Path $resultRoot 'report.md') -Encoding UTF8

    if ($events.Count -gt 0 -and -not $NoQueue) {
        $comparisonInstruction = if ($Family -eq 'control') {
            '与 pairhist_v2 相同训练种子和更新轮次的验证结果配对比较，同时检查旧 dminj30；'
        } else {
            '与旧 dminj 对照；'
        }
        $message = 'QKD服务器' + $Family + '训练出现新里程碑。结果已同步到 ' + $resultRoot +
            '；事件：' + ($events -join '；') +
            '。请读取 status.json、metrics.jsonl 和训练日志；检查 KL、裁剪率、Critic 和验证集，' +
            $comparisonInstruction + '若发现明确可修的问题，在服务器隔离分支修改并验证，再决定后续训练；' +
            '不要覆盖本地主项目。仅在完成、失败或需要采取行动时通知用户。'
        & 'D:\nvm\nodejs\codex.cmd' queue --thread $threadId --message $message
        if ($LASTEXITCODE -ne 0) { throw "codex queue failed: exit=$LASTEXITCODE" }
        Write-PollLog "Queued: $($events -join '; ')"
    } else {
        Write-PollLog "Polled; updates=$((@($status.runs | ForEach-Object latest_update)) -join ','); events=$($events.Count)"
    }
    $previous | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $signalFile -Encoding UTF8
} catch {
    Write-PollLog "ERROR: $($_.Exception.Message)"
    throw
}
