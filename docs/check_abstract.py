"""Abstract kelime sayısı ve main.tex özeti."""
import re, sys
from pathlib import Path

tex_path = Path(__file__).parent.parent / "paper" / "main.tex"
content = tex_path.read_text(encoding="utf-8")

# Abstract ayıkla
m = re.search(r"\\abstract\{%(.+?)\n\}", content, re.DOTALL)
if m:
    raw = m.group(1)
    clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r" \1 ", raw)
    clean = re.sub(r"\\[a-zA-Z]+", " ", clean)
    clean = re.sub(r"[${}~\[\]\\]", " ", clean)
    words = [w for w in clean.split() if w]
    n = len(words)
    status = "OK" if n <= 200 else f"ASIM (+{n-200})"
    print(f"Abstract: {n} kelime [{status}]")
else:
    print("Abstract bulunamadi")

lines = content.count("\n") + 1
print(f"main.tex: {lines} satir")

# Title kontrolü
tm = re.search(r"\\Title\{(.+?)\}", content, re.DOTALL)
if tm:
    t = " ".join(tm.group(1).split())
    print(f"Baslik: {t[:80]}...")
