# deploy.ps1 - Git & GitHub Pages deployment configuration helper

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       CAR PRICE PREDICTOR FRONTEND DEPLOYMENT HELPER      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Rename default branch to 'main'
git branch -m main
Write-Host "[OK] Local branch renamed to 'main'" -ForegroundColor Green

# 2. Add remote origin if not present
$existingRemotes = git remote
$hasOrigin = $false
foreach ($r in $existingRemotes) {
    if ($r -eq "origin") {
        $hasOrigin = $true
    }
}

if ($hasOrigin) {
    Write-Host "[Info] Remote 'origin' is already configured." -ForegroundColor Yellow
} else {
    git remote add origin "https://github.com/areef2297-JD/car-price-predictor.git"
    Write-Host "[OK] Remote 'origin' set to: https://github.com/areef2297-JD/car-price-predictor.git" -ForegroundColor Green
}

Write-Host ""
Write-Host "To push and deploy your project, please follow these steps:" -ForegroundColor White
Write-Host "----------------------------------------------------------" -ForegroundColor Gray

Write-Host "Step 1: Create the repository on GitHub" -ForegroundColor Yellow
Write-Host "  * Open your browser and go to: https://github.com/new"
Write-Host "  * Set the Repository Name to: car-price-predictor"
Write-Host "  * Keep it Public, leave 'Add a README' unchecked, and click 'Create Repository'."

Write-Host ""
Write-Host "Step 2: Push your local commit to GitHub" -ForegroundColor Yellow
Write-Host "  * Run this command in your terminal:"
Write-Host "    git push -u origin main" -ForegroundColor Cyan
Write-Host "  * (If prompted, log in via your browser or paste your GitHub Personal Access Token)."

Write-Host ""
Write-Host "Step 3: Enable GitHub Pages hosting" -ForegroundColor Yellow
Write-Host "  * Go to: https://github.com/areef2297-JD/car-price-predictor/settings/pages"
Write-Host "  * Under 'Build and deployment' -> 'Source', select 'Deploy from a branch'."
Write-Host "  * Under 'Branch', select 'main' (and '/ (root)') and click 'Save'."
Write-Host "  * In a few moments, your web page will be live at:"
Write-Host "    https://areef2297-jd.github.io/car-price-predictor/" -ForegroundColor Green
Write-Host "----------------------------------------------------------" -ForegroundColor Gray
