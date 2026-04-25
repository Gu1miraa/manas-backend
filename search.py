from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from parser import load_all_documents
import numpy as np
import time
import os

print("--- [Человек 3] Инициализация мультиязычной системы поиска ---")

# 1. Используем мощную мультиязычную модель (Русский <-> Турецкий <-> Английский)
# Она чуть тяжелее, но гораздо лучше связывает смыслы разных языков
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

# 2. Загружаем базу данных
docs_path = "docs"
if not os.path.exists(docs_path):
    os.makedirs(docs_path)
    print(f"(!) Папка {docs_path} была пуста. Положи туда PDF и TXT файлы!")

print(f"--- Загрузка документов из папки {docs_path} ---")
knowledge_base = load_all_documents(docs_path)

if not knowledge_base:
    print("(!) ОШИБКА: Документы не найдены. Проверь папку docs.")
    texts = []
    texts_emb = []
else:
    texts = [item["text"] for item in knowledge_base]
    print(f"--- Найдено {len(texts)} фрагментов. Начинаю индексацию... ---")
    
    # ПРЕВРАЩАЕМ ТЕКСТ В ВЕКТОРЫ
    start_time = time.time()
    # show_progress_bar поможет видеть, что процесс идет
    texts_emb = model.encode(texts, show_progress_bar=True)
    end_time = time.time()
    
    print(f"--- Индексация завершена за {end_time - start_time:.2f} сек. ---")

def search_answer(question: str):
    if not texts:
        return "База данных пуста."

    question_emb = model.encode([question])
    scores = cosine_similarity(question_emb, texts_emb)[0]
    
    # Берем индексы 3-х самых похожих фрагментов
    top_indices = np.argsort(scores)[-3:][::-1] 
    
    # Проверяем, есть ли среди них очень хорошие совпадения
    best_idx = top_indices[0]
    best_score = scores[best_idx]

    if best_score < 0.25: # Снизим порог для мультиязычности
        return "К сожалению, я не нашел точного ответа в документах."

    # Собираем ответ из лучшего фрагмента
    return {
        "text": texts[best_idx],
        "source": knowledge_base[best_idx]["source"],
        "score": round(float(best_score), 2)
    }

# ТЕСТОВЫЙ ИНТЕРФЕЙС
if __name__ == "__main__":
    print("\n" + "="*50)
    print("СИСТЕМА ГОТОВА. Теперь я понимаю русский и турецкий!")
    print("="*50)
    
    while True:
        query = input("\nВведите вопрос (или 'exit' для выхода): ")
        if query.lower() == 'exit': 
            break
        
        result = search_answer(query)
        
        if isinstance(result, str):
            print(f"Результат: {result}")
        else:
            print(f"\n[НАЙДЕННЫЙ ФРАГМЕНТ]:\n{result['text']}")
            print(f"\n[ИСТОЧНИК]: {result['source']} | [УВЕРЕННОСТЬ]: {result['score']}")
            print("-" * 30)