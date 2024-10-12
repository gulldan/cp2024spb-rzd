import asyncio
import logging
import os
import random
import re
import time
import urllib.parse

import aiofiles
import polars as pl
import ujson as json
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from rapidfuzz import fuzz

# Максимальная длинна колонки для фильрации добавления в запрос
MAX_LEN = 25

# Количество запросов к топу поисковой выдаче
NUM_RESULTS = 7

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

urls = [
    "https://www.vseinstrumenti.ru",
    "https://www.chipdip.ru",
    "https://market.yandex.ru",
    "https://lemanapro.ru",
    "https://technoavia.ru",
    "https://www.komus.ru",
    "https://www.citilink.ru",
    "https://www.abrasives.ru",
    "https://attacher.ru",
    "https://www.220-volt.ru",
    "https://www.xcom-shop.ru",
    "https://www.sima-land.ru",
    "https://bi-bi.ru",
]

# Список ключевых слов для поиска характеристик в тексте
characteristic_keywords = [
    "Характеристики",
    "Коротко о товаре",
    "Основные характеристики",
    "Технические характеристики",
    "Подробные характеристики",
    "Описание",
]

file_lock = asyncio.Lock()  # Глобальный лок для операций с файлами


async def extract_structured_data(page, search_query):  # noqa: C901
    """Извлекает структурированные данные из текста страницы.

    Args:
        page: Страница Playwright
        search_query: Поисковый запрос для дополнительного контекста

    Returns:
        Данные в структурированном виде
    """
    logging.info("Извлечение структурированных данных...")
    start_time = time.time()

    structured_data = {}

    try:
        # Получаем весь контент страницы
        page_content = await page.content()

        # Ищем JSON внутри контента страницы
        matches = re.findall(r"\{.*?\}", page_content)
        logging.debug(f"Найдено {len(matches)} потенциальных JSON блоков.")
        for match in matches:
            try:
                data = json.loads(match)
                # Проверяем, содержит ли JSON нужные данные
                if isinstance(data, dict) and "name" in data and "items" in data:
                    structured_data = data
                    break
            except:  # noqa: E722, S112
                continue

        if not structured_data:
            # Если не нашли JSON, пробуем получить текстовые данные
            elements = await page.query_selector_all("div, table")
            logging.debug(f"Количество элементов для проверки: {len(elements)}")
            for element in elements:
                text = await element.inner_text()
                for keyword in characteristic_keywords:
                    if keyword in text:
                        # Обработка текста для извлечения характеристик
                        characteristics = {}
                        lines = text.split("\n")
                        for line in lines:
                            if ": " in line:
                                key, value = line.split(": ", 1)
                                characteristics[key.strip()] = value.strip()
                        if characteristics:
                            structured_data["Характеристики"] = characteristics
                            break
                if structured_data:
                    break

    except Exception as e:
        logging.exception(f"Ошибка при извлечении данных: {e}")

    end_time = time.time()
    logging.debug(f"Время извлечения структурированных данных: {end_time - start_time:.2f} секунд")
    return structured_data


async def process_link(context, link, search_query, idx, code):
    """Обрабатывает отдельную ссылку и возвращает извлеченные данные."""
    page = await context.new_page()
    logging.info(f"Переход по результату {idx + 1}: {link}")
    try:
        await page.goto(link, wait_until="domcontentloaded", timeout=30000)

        try:
            logging.info("Ожидание загрузки новой страницы...")
            await page.wait_for_load_state("load", timeout=15000)
            await page.wait_for_selector("body", state="visible", timeout=5000)

            # Имитация скроллинга
            await page.mouse.wheel(0, 500)
            await asyncio.sleep(random.uniform(1, 2))
            await page.mouse.wheel(0, -500)
            await asyncio.sleep(random.uniform(1, 2))

            # Щелчок по случайному элементу для имитации взаимодействия
            await page.click("body")
            await asyncio.sleep(random.uniform(1, 2))

        except Exception as e:
            logging.exception(f"Ошибка при взаимодействии со страницей: {e}")

        # Извлечение структурированных данных с страницы
        structured_data = await extract_structured_data(page, search_query)

        # Извлечение заголовка страницы
        name = await page.title()
        result = {"name": name, "link": link, "structured_data": structured_data}

        # Сохранение результатов в JSON с использованием замка
        filename = f"{code}_webpage_result_{idx + 1}.json"
        async with file_lock, aiofiles.open(filename, "w", encoding="utf-8") as json_file:
            await json_file.write(json.dumps(result, ensure_ascii=False, indent=4))

        logging.info(f"Результаты сохранены для {link} в {filename}.")

        await page.close()
        return structured_data.get("Характеристики", {})
    except Exception as e:
        logging.exception(f"Ошибка при обработке ссылки {link}: {e}")
        await page.close()
        return {}


def merge_characteristics(characteristics_list, threshold=75):
    """Объединяет характеристики из списка, используя rapidfuzz.

    Args:
        characteristics_list: Список словарей с характеристиками.
        threshold: Пороговое значение для схожести характеристик.

    Returns:
        Объединенный словарь характеристик.
    """
    merged_characteristics = {}
    for characteristics in characteristics_list:
        for key, value in characteristics.items():
            found = False
            for m_key in merged_characteristics:
                similarity = fuzz.ratio(key, m_key)
                if similarity >= threshold:
                    merged_characteristics[m_key] = value
                    found = True
                    break
            if not found:
                merged_characteristics[key] = value
    return merged_characteristics


async def scrape_search_results(search_query: str, code: str, num_results: int = 7):  # noqa: C901, PLR0915
    """Выполняет поиск по Google по списку сайтов и собирает контент первых результатов n.

    Args:
        search_query (str): Поисковый запрос.
        code (str): Уникальный код для именования файлов, код МТР.
        num_results (int, optional): Количество ссылок, которые будут собраны. По умолчанию 7.

    Returns:
        Объединенные характеристики.
    """
    total_start_time = time.time()

    # Запуск браузера
    stealth = Stealth(init_scripts_only=True)
    try:
        async with async_playwright() as p:
            browser_start_time = time.time()
            browser = await p.chromium.launch(headless=True)
            browser_end_time = time.time()
            logging.debug(f"Время запуска браузера: {browser_end_time - browser_start_time:.2f} секунд")

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",  # noqa: E501
                locale="ru-RU",
            )
            # Включаем stealth режим
            await stealth.apply_stealth_async(context)

            page = await context.new_page()
            # Дополнительные HTTP-заголовки
            await page.set_extra_http_headers(
                {
                    "Referer": "https://www.google.com/",
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Cache-Control": "max-age=0",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "same-origin",
                    "Sec-Fetch-User": "?1",
                }
            )

            async def get_links(search_query, use_site_filters=True):
                # Формируем строку для поиска по нескольким сайтам
                if use_site_filters:
                    site_query = " OR ".join([f"site:{url}" for url in urls])
                    full_query = f"{site_query} {search_query}"
                else:
                    full_query = search_query

                # Кодирование запроса для URL
                term = urllib.parse.quote_plus(full_query)
                url = f"https://www.google.com/search?q={term}"

                # Рандомная задержка для имитации поведения человека
                await asyncio.sleep(random.uniform(1, 2))

                logging.info(f"Переход по URL поиска: {url}")
                navigation_start_time = time.time()
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                except PlaywrightTimeoutError:
                    logging.warning(f"Таймаут при переходе по URL поиска: {url}")
                    return []

                navigation_end_time = time.time()
                logging.debug(f"Время загрузки страницы поиска: {navigation_end_time - navigation_start_time:.2f} секунд")

                try:
                    logging.info("Ожидание загрузки результатов поиска...")
                    wait_start_time = time.time()
                    await page.wait_for_selector("h3", state="visible", timeout=10000)
                    wait_end_time = time.time()
                    logging.debug(f"Время ожидания результатов поиска: {wait_end_time - wait_start_time:.2f} секунд")
                except Exception as e:
                    logging.warning(f"Не удалось дождаться загрузки результатов поиска: {e}")
                    return []

                links = []

                logging.info(f"Извлечение первых {num_results} ссылок...")
                extraction_start_time = time.time()
                for i in range(num_results):
                    try:
                        link = await page.locator("h3").nth(i).evaluate("el => el.closest('a')?.href")
                        if link:
                            links.append(link)
                    except Exception as e:
                        logging.exception(f"Ошибка при извлечении ссылки {i + 1}: {e}")
                        break
                extraction_end_time = time.time()
                logging.debug(f"Время извлечения ссылок: {extraction_end_time - extraction_start_time:.2f} секунд")
                return links

            # Получаем ссылки с site-specific фильтрами
            links = await get_links(search_query, use_site_filters=True)

            # Если не нашли результаты, пробуем поискать без site-specific фильтра
            if not links:
                logging.warning("Не найдено результатов поиска по заданным сайтам. Выполняем общий поиск.")
                links = await get_links(search_query, use_site_filters=False)

            await page.close()

            if not links:
                logging.error("Не найдено результатов поиска даже при общем поиске.")
                await browser.close()
                return None

            characteristics_list = []

            # Обработка ссылок параллельно с ограничением времени
            tasks = []
            for idx, link in enumerate(links):
                task = asyncio.create_task(process_link(context, link, search_query, idx, code))
                tasks.append(task)

            # Собираем результаты с таймаутом
            characteristics_list = []
            for task in asyncio.as_completed(tasks, timeout=60):
                try:
                    characteristics = await asyncio.wait_for(task, timeout=60)
                    if characteristics:
                        characteristics_list.append(characteristics)
                except TimeoutError:
                    logging.warning("Таймаут при обработке ссылки.")
                except Exception as e:
                    logging.exception(f"Ошибка при обработке ссылки: {e}")

            # Объединяем характеристики с использованием rapidfuzz
            merged_characteristics = merge_characteristics(characteristics_list, threshold=75)

            # Удаление промежуточных файлов
            for idx in range(len(links)):
                filename = f"{code}_webpage_result_{idx + 1}.json"
                try:
                    os.remove(filename)
                    logging.info(f"Удален файл {filename}.")
                except Exception as e:
                    logging.warning(f"Не удалось удалить файл {filename}: {e}")

            logging.info("Закрытие браузера...")
            browser_close_start_time = time.time()
            await browser.close()
            browser_close_end_time = time.time()
            logging.debug(f"Время закрытия браузера: {browser_close_end_time - browser_close_start_time:.2f} секунд")

    except Exception as e:
        logging.exception(f"Глобальная ошибка при выполнении scrape_search_results: {e}")
        return None

    total_end_time = time.time()
    logging.debug(f"Общее время выполнения скрипта: {total_end_time - total_start_time:.2f} секунд")

    return merged_characteristics


async def main():
    # Читаем данные из '../data/MTR.parquet' с помощью polars
    df = pl.read_parquet("../data/MTR.parquet")

    # Фильтруем строки, где 'Параметры' пустые
    df_filtered = df.filter(pl.col("Параметры").is_null() | (pl.col("Параметры") == ""))

    # Создаем папку output, если она не существует
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    # Обходим строки последовательно
    for row in df_filtered.iter_rows(named=True):
        name = row.get("Наименование", "")
        name = name.strip() if name is not None else ""
        marking = row.get("Маркировка", "")
        marking = marking.strip() if marking is not None else ""
        code = row.get("код СКМТР", "")
        code = "" if code is None else str(code).strip()

        output_file = os.path.join(output_folder, f"{code}.json")
        if os.path.exists(output_file):
            logging.info(f"Файл {output_file} уже существует. Пропускаем.")
            continue

        # Проверяем длину полей
        name_len = len(name)
        marking_len = len(marking)

        # Определяем поисковый запрос согласно условиям
        search_query = f"{name} {marking}".strip() if name_len <= MAX_LEN and marking_len <= MAX_LEN else marking

        if not search_query:
            logging.warning(f"Пустой поисковый запрос для кода {code}")
            # Создаем пустой файл, чтобы не искать снова
            async with aiofiles.open(output_file, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps({}, ensure_ascii=False, indent=4))
            continue

        logging.info(f"Начинаем обработку для кода {code} с запросом '{search_query}'")

        # Выполняем поиск и извлечение характеристик с таймаутом
        try:
            merged_characteristics = await asyncio.wait_for(scrape_search_results(search_query, code, NUM_RESULTS), timeout=180)
        except TimeoutError:
            logging.warning(f"Таймаут при обработке кода {code}")
            merged_characteristics = None
        except Exception as e:
            logging.exception(f"Ошибка при обработке кода {code}: {e}")
            merged_characteristics = None

        if merged_characteristics is not None:
            # Сохраняем объединенные характеристики в JSON-файл с названием из 'код СКМТР'
            async with aiofiles.open(output_file, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(merged_characteristics, ensure_ascii=False, indent=4))
            logging.info(f"Характеристики для кода {code} сохранены в {output_file}")
        else:
            logging.warning(f"Не удалось получить характеристики для кода {code}")
            # Создаем пустой файл, чтобы не искать снова
            async with aiofiles.open(output_file, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps({}, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
