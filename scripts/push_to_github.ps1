<#
PowerShell helper to push the current repo to a GitHub remote.

Usage (PowerShell, from project root):
  .\scripts\push_to_github.ps1 -RemoteUrl 'https://github.com/Emomeow/Undergraduate-Thesis.git' -Branch 'main' -CommitMsg 'Sync: add cheatsheet'

The script will:
 - check that `git` is available
 - show current branch and status
 - stage & commit if there are uncommitted changes
 - add or update remote `origin` to the provided URL
 - push the current branch and set upstream

Note: run this locally — the agent cannot run Git in its environment.
#>
param(
    [string]$RemoteUrl = "https://github.com/Emomeow/Undergraduate-Thesis.git",
    [string]$Branch = "main",
    [string]$CommitMsg = "Sync: update README and developer cheatsheet",
    [switch]$UseSsh
)

function ExitWith([string]$msg){ Write-Host $msg -ForegroundColor Red; exit 1 }

# check git
if (-not (Get-Command git -ErrorAction SilentlyContinue)){
    ExitWith "git not found in PATH. Install Git for Windows: https://git-scm.com/download/win and retry."
}

Write-Host "Git found. Showing repo status..." -ForegroundColor Cyan
$branch = git rev-parse --abbrev-ref HEAD 2>$null
if (-not $branch){ ExitWith "Not a git repository (no branch detected). Run 'git init' or open the repository root." }
Write-Host "Current branch: $branch"

$porcelain = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($porcelain)){
    Write-Host "Uncommitted changes detected. Staging and committing with message:`n  $CommitMsg" -ForegroundColor Yellow
    git add -A
    git commit -m "$CommitMsg"
} else {
    Write-Host "No changes to commit." -ForegroundColor Green
}

# set remote
if ($UseSsh) {
    $RemoteUrlToUse = $RemoteUrl -replace '^https://github.com/', 'git@github.com:'
} else {
    $RemoteUrlToUse = $RemoteUrl
}

if (git remote get-url origin 2>$null) {
    Write-Host "Updating existing remote 'origin' to $RemoteUrlToUse"
    git remote set-url origin $RemoteUrlToUse
} else {
    Write-Host "Adding remote 'origin' -> $RemoteUrlToUse"
    git remote add origin $RemoteUrlToUse
}

function AttemptPush {
    param([string]$Branch)
    Write-Host "Pushing branch '$Branch' to origin (upstream set)..." -ForegroundColor Cyan
    git push -u origin $Branch
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Push succeeded." -ForegroundColor Green
        return $true
    } else {
        Write-Host "Initial push failed (exit code $LASTEXITCODE)." -ForegroundColor Yellow
        return $false
    }
}

if (-not (AttemptPush -Branch $branch)) {
    # Detect non-fast-forward (behind) and attempt automatic rebase
    Write-Host "Attempting to auto-resolve non-fast-forward (fetch + rebase) ..." -ForegroundColor Cyan
    git fetch origin $branch
    if ($LASTEXITCODE -ne 0) { ExitWith "Fetch failed; resolve manually (check network/permissions)." }
    # Prefer rebase to keep linear history
    git pull --rebase origin $branch
    if ($LASTEXITCODE -ne 0) { ExitWith "Rebase failed; manual conflict resolution required. Run 'git status' and fix conflicts before re-running script." }
    Write-Host "Rebase completed; retrying push..." -ForegroundColor Cyan
    if (-not (AttemptPush -Branch $branch)) {
        ExitWith "Push still failing. Inspect output above; you may need to resolve conflicts or authentication issues." }
}

Write-Host "If push required authentication, you may be prompted for credentials or to use a PAT. For a smoother flow, consider setting up SSH keys and using the SSH URL." -ForegroundColor Yellow

Write-Host "If push required authentication, you may be prompted for credentials or to use a PAT. For a smoother flow, consider setting up SSH keys and using the SSH URL." -ForegroundColor Yellow
