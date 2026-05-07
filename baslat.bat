@echo off
setlocal
title Celik Kubbe - TEKNOFEST 2026
cd /d "%~dp0"

echo [CELIK KUBBE] Uygun Python ortami araniyor...

set PYTHON_PATH=python
if exist "C:\Users\Victus\miniconda3\python.exe" (
    set PYTHON_PATH=C:\Users\Victus\miniconda3\python.exe
    echo [OK] Miniconda Python bulundu: %PYTHON_PATH%
) else (
    echo [UYARI] Miniconda bulunamadi, varsayilan python kullanilacak.
)

echo.
echo [CELIK KUBBE] Sistem ortami:
%PYTHON_PATH% --version
echo Path: %PYTHON_PATH%
echo.

echo [CELIK KUBBE] PySide6 arayuz baslatiliyor...
%PYTHON_PATH% main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo HATA: Uygulama baslatilamadi. (Hata Kodu: %ERRORLEVEL%)
    if %ERRORLEVEL% equ -1073741819 (
        echo [!] Bellek hatasi (Access Violation). Yanlis Python surumu kullaniliyor olabilir.
    )
    echo Python veya bagimliliklar eksik olabilir.
    echo Gereksinimler: pip install -r requirements.txt
    pause
)
endlocal
