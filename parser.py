import os
import fitz  # PyMuPDF для PDF
import docx  # для DOCX
import pandas as pd  # для Excel
import warnings

# Игнорируем предупреждения openpyxl при чтении стилей Excel
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

def read_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def read_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def read_excel(path):
    # Читаем все листы и превращаем каждую строку в текстовое описание
    text_data = []
    df_dict = pd.read_excel(path, sheet_name=None)
    for sheet_name, df in df_dict.items():
        # Заполняем пустые ячейки, чтобы не было ошибок
        df = df.fillna("")
        for _, row in df.iterrows():
            # Превращаем строку таблицы в текст: "Колонка: Значение, Колонка2: Значение"
            row_str = f"Лист {sheet_name}: " + ", ".join([f"{col}: {val}" for col, val in row.items()])
            text_data.append(row_str)
    return "\n".join(text_data)

def load_all_documents(folder_path="docs"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"(!) Папка '{folder_path}' создана. Положите туда файлы.")
        return []

    all_chunks = []
    print(f"--- Начинаю загрузку документов из '{folder_path}' ---")

    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        ext = filename.lower()
        content = ""

        try:
            if ext.endswith(".pdf"):
                content = read_pdf(path)
                print(f"[OK] Прочитан PDF: {filename}")
            elif ext.endswith(".docx"):
                content = read_docx(path)
                print(f"[OK] Прочитан Word: {filename}")
            elif ext.endswith(".xlsx") or ext.endswith(".xls"):
                content = read_excel(path)
                print(f"[OK] Прочитан Excel: {filename}")
            elif ext.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                print(f"[OK] Прочитан TXT: {filename}")

            if content:
                # Нарезка на фрагменты (chunks)
                # Для таблиц (Excel) мы уже сделали нарезку по строкам, 
                # для текста делим по абзацам.
                paragraphs = [p.strip() for p in content.split("\n") if len(p.strip()) > 25]
                for p in paragraphs:
                    all_chunks.append({"text": p, "source": filename})
        
        except Exception as e:
            print(f"[ERROR] Не удалось прочитать {filename}: {e}")

    print(f"--- Загрузка завершена. Всего фрагментов: {len(all_chunks)} ---")
    return all_chunks

if __name__ == "__main__":
    # Тестовый запуск только для Человека 2
    data = load_all_documents("docs")
    if data:
        print("\nПример данных (первые 2 фрагмента):")
        for i in range(min(2, len(data))):
            print(f"- {data[i]['text'][:100]}... [Из: {data[i]['source']}]")