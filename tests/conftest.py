"""Pytest bootstrap: fix conda/matplotlib libiomp5md.dll duplication on Windows."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
