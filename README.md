# AI Economist Budgeting Service

Сервис предоставляет REST API для загрузки бюджетов по товарным категориям и
анализа таблиц закупок относительно доступных лимитов. Он заменяет прежнюю
симуляцию микроэкономики на утилиту, которая помогает экономисту быстро
соотнести покупки с бюджетными категориями и понять, достаточно ли средств по
каждой группе.

## Возможности

- Загрузка бюджета в виде набора строк с названием категории, лимитом и
  ключевыми словами, которые помогают автоматически сопоставлять покупки.
- Получение текущего состояния бюджета и суммарного лимита.
- Анализ таблицы закупок: сервис распределяет позиции по категориям (используя
  подсказку `category_hint` или совпадение по ключевым словам), рассчитывает
  потребность и остаток и возвращает отчёт о достаточности средств.

## Структура проекта

- **`service/api`** — приложение FastAPI с эндпоинтами `/budget` и
  `/purchases/analyze`.
- **`service/budget.py`** — in-memory менеджер бюджета с логикой сопоставления
  покупок и расчёта итоговых сумм по категориям.
- **`tests/`** — модульные тесты для API и бизнес-логики.
- **`scripts/`** — вспомогательные скрипты для локального запуска.

## Быстрый старт

1. Убедитесь, что Ollama запущен отдельно от этого проекта и доступен по
   адресу `http://localhost:11434` (например, при помощи сервиса из примера
   ниже). Сервис FastAPI обращается к Ollama через переменную окружения
   `OLLAMA_HOST`.
2. Соберите и поднимите сервис:

После сборки сервис доступен по адресу `http://localhost:8081`. FastAPI
документация будет доступна на `http://localhost:8081/docs`.

### Проверка работы сервиса

После запуска приложения можно быстро убедиться, что оно отвечает, при помощи
health-check эндпоинта:

```bash
curl http://localhost:8081/health
```

В ответ должен прийти JSON `{"status": "ok"}`. Для автоматической проверки
всей бизнес-логики выполните тесты:

```bash
pytest
```

### Отдельный сервис Ollama

Пример docker-compose для запуска Ollama с моделью:

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    depends_on:
      gpu-check:
        condition: service_completed_successfully
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_ORIGINS=*
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ollama:/root/.ollama
    gpus: all
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 10

  init-model:
    image: ollama/ollama:latest
    depends_on:
      ollama:
        condition: service_healthy
    environment:
      - OLLAMA_HOST=http://ollama:11434
    command: ["/bin/sh","-lc","ollama pull krith/qwen2.5-32b-instruct:IQ4_XS"]
    restart: "no"

volumes:
  ollama:
```

В примере предполагается, что сервис `gpu-check` уже описан в том же файле и
проверяет доступность GPU перед стартом Ollama.

При запуске `docker compose up` для этого проекта контейнер FastAPI будет
обращаться к Ollama по адресу `http://host.docker.internal:11434`. При
необходимости можно переопределить переменную `OLLAMA_HOST` в
`docker-compose.yml` или задать её при запуске.

## Работа с API

### Отправка файла на анализ

Эндпоинт `/llm/purchases` принимает полный текст договора и извлекает из него
таблицу со спецификацией. Можно передать файл с текстом договора напрямую из
CLI, например, файл `econom.txt`:

```bash
curl -X POST http://localhost:8081/llm/purchases \
  -H "Content-Type: multipart/form-data" \
  -F "file=@econom.txt"
```

Ответ содержит JSON со списком товаров, ценами, количеством, суммами и
определёнными категориями бюджета.

### Загрузка бюджета

```bash
curl -X POST http://localhost:8081/budget \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {"category": "Электроника", "limit": 2000000, "keywords": ["ноутбук", "монитор"]},
      {"category": "Крупная бытовая техника", "limit": 5215000, "keywords": ["холодильник"]},
      {"category": "Оргтехника", "limit": 600000, "keywords": ["принтер", "сканер"]}
    ]
  }'
```

Ответ содержит список категорий с лимитами и суммарный бюджет. Текущий бюджет
можно получить повторным GET-запросом на `/budget`.

### Анализ таблицы закупок

```bash
curl -X POST http://localhost:8081/purchases/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "purchases": [
      {"description": "Ноутбук Dell", "amount": 1200000, "category_hint": null},
      {"description": "Холодильник офисный", "amount": 500300, "category_hint": "Крупная бытовая техника"},
      {"description": "МФУ для отдела", "amount": 750000, "category_hint": null}
    ]
  }'
```

В ответе сервис перечислит распознанные категории, укажет, где средств хватает
или не хватает, и вернёт распределение позиций по категориям вместе с остатками.
Нераспознанные покупки выводятся отдельно, чтобы их можно было обработать
вручную.

## Тестирование

```bash
pytest
```

## Лицензия

Проект распространяется по лицензии MIT.
