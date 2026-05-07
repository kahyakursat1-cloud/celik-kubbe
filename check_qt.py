import sys
try:
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    print("[OK] PySide6 QApplication initialized successfully.")
    app.quit()
except Exception as e:
    print(f"[FAIL] PySide6 initialization failed: {e}")
    sys.exit(1)
