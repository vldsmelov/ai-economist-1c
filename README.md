# AI Economist Service

Эта репозитория содержит полностью автономную реализацию сервиса `ai-economist`,
вынесенную из монорепозитория `combine_1cdo_autoaccept`. Код объединяет модель
микроэкономической среды, простую прогрессивную налоговую политику и
веб-сервис для запуска симуляций.

## Основные компоненты

* **`ai_economist_service.environment`** — упрощённое грид-окружение с агентами,
  которые перемещаются, добывают ресурсы и зарабатывают доход.
* **`ai_economist_service.planner`** — реализация прогрессивной налоговой системы
  с перераспределением собранных средств.
* **`ai_economist_service.simulation`** — единая точка оркестрации, объединяющая
  окружение и планировщика, а также предоставляющая удобный API для запуска
  эпизодов и сериализации их результатов.
* **`ai_economist_service.api`** — REST API на базе FastAPI, которое предоставляет
  эндпоинт `/simulate` для запуска эпизода по входной конфигурации.
* **`scripts/run_episode.py`** — CLI-утилита для запуска симуляции из командной
  строки.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn ai_economist_service.api.app:app --reload
```

После запуска сервиса отправьте POST-запрос на `/simulate`:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"steps": 20, "seed": 123}'
```

## Тестирование

```bash
pytest
```

## Лицензия

Проект распространяется по лицензии MIT.
