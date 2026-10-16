@echo off
echo ============================================
echo   SmartFarmingAI - Starting Server
echo ============================================
cd /d c:\Users\RISHAV\.gemini\antigravity\scratch\SmartFarmingAI
call venv\Scripts\activate.bat
echo.
echo Checking model weights...
if exist saved_models\best_model.pth (
    echo [OK] Found: saved_models\best_model.pth
) else (
    echo [WARN] No trained weights found. Will use raw ViT pretrained weights.
    echo        For best results, train the model first and save to saved_models\best_model.pth
)
echo.
echo Starting FastAPI server on http://0.0.0.0:8000
echo Access from another computer: http://<YOUR-IP>:8000
echo.
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /i "IPv4"') do (
    for /f "tokens=1" %%j in ("%%i") do (
        echo Your local IP: %%j
        echo Other devices: http://%%j:8000
    )
)
echo.
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
pause
