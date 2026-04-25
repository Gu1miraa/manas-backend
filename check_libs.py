import sys

def check_setup():
    print("--- Проверка системы ---")
    print(f"Версия Python: {sys.version}")
    
    libraries = [
        "sentence_transformers",
        "sklearn",
        "numpy",
        "pymupdf", 
        "docx",
        "pandas"
    ]
    
    missing = []
    for lib in libraries:
        try:
            if lib == "pymupdf":
                import fitz
            else:
                __import__(lib)
            print(f"[V] {lib} — Установлено")
        except ImportError:
            print(f"[X] {lib} — НЕ НАЙДЕНО")
            missing.append(lib)
    
    if missing:
        print("\nНужно доустановить:")
        # Специально для fitz пишем pymupdf
        to_install = ["pymupdf" if m == "pymupdf" else m for m in missing]
        print(f"pip install {' '.join(to_install)}")
    else:
        print("\nВСЁ ГОТОВО! Можно запускать search.py")

if __name__ == "__main__":
    check_setup()
    