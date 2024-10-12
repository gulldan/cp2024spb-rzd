**Описание работы скрипта парсинга параметров**

---

### **Бизнесовая точка зрения**

Данный скрипт предназначен для автоматизированного сбора и обогащения данных о товарах, у которых отсутствуют параметры в базе данных. Бизнес-цель скрипта — заполнить пробелы в информации о товарах, используя данные из интернета, что позволяет:

- Улучшить качество и полноту информации о товарах в системе.
- Повысить удовлетворенность клиентов за счет предоставления более детальной информации.
- Сократить ручной труд, автоматизируя процесс сбора данных.
- Повысить конкурентоспособность за счет более информативного каталога товаров.

---

### **Техническая точка зрения**

Скрипт выполняет следующие основные задачи:

1. **Чтение и фильтрация данных из файла `MTR.parquet`**:

   - Используется библиотека **Polars** для эффективного чтения и обработки больших объемов данных.
   - Из исходного файла `MTR.parquet` выбираются записи, где поле `'Параметры'` (`'Parameters'`) пустое или отсутствует. Это обеспечивает фокусировку на товарах, для которых требуется дополнить информацию.

2. **Формирование поисковых запросов**:

   - Для каждого отфильтрованного товара извлекаются поля `'Наименование'` (`'Name'`), `'Маркировка'` (`'Marking'`) и `'код СКМТР'` (`'Code'`).
   - В зависимости от длины этих полей формируется поисковый запрос:
     - Если длина `'Наименование'` ≤ 25 символов и `'Маркировка'` ≤ 25 символов, запрос состоит из объединения этих двух полей.
     - В остальных случаях используется только `'Маркировка'`.
   - Это условие позволяет оптимизировать поисковый запрос для получения более релевантных результатов.

3. **Выполнение веб-поиска и сбор ссылок**:

   - **Playwright** и **asyncio** используются для асинхронного управления браузером без интерфейса пользователя.
   - Скрипт выполняет поиск в Google с использованием сформированного запроса.
     - Первоначально поиск ограничен списком определенных сайтов (например, `"site:https://www.vseinstrumenti.ru"`), чтобы получить более целевые результаты.
     - Если результатов нет, выполняется общий поиск без ограничения по сайтам.
   - Собираются ссылки на первые несколько результатов поиска (по умолчанию 7).

4. **Обход и обработка собранных ссылок**:

   - Каждая ссылка обрабатывается асинхронно, что ускоряет процесс.
   - При посещении каждой страницы:
     - Эмулируется поведение пользователя (прокрутка, случайные клики), чтобы избежать блокировки со стороны сайтов.
     - Извлекается контент страницы.
     - Пытается найти структурированные данные (JSON) в контенте страницы.
     - Если структурированные данные не найдены, производится поиск разделов с ключевыми словами, такими как `"Характеристики"`, `"Описание"` и т.д.
     - Извлекаются характеристики товара в виде пар "ключ-значение".
   - Результаты по каждой ссылке сохраняются во временные JSON-файлы с уникальными именами, включающими `'код СКМТР'`.

5. **Объединение и обработка характеристик**:

   - Используется библиотека **rapidfuzz** для объединения характеристик из разных источников, учитывая возможные вариации в названиях характеристик.
   - Алгоритм сопоставляет похожие ключи (названия характеристик) с помощью функции похожести строк и объединяет их, чтобы получить единый набор характеристик для товара.

6. **Сохранение результатов**:

   - Объединенные характеристики сохраняются в файл JSON в папке `output`, где имя файла соответствует `'код СКМТР'`.
   - Если характеристики не были найдены или возникла ошибка, создается пустой файл JSON, чтобы в будущем не повторять попытку для этого товара.
   - После обработки ссылки временные файлы удаляются для экономии места и избежания конфликтов при параллельной работе.

7. **Обработка ошибок и обеспечение надежности**:

   - Реализованы механизмы обработки исключений и таймаутов для предотвращения зависаний скрипта.
   - Если при обработке ссылки или поискового запроса возникает ошибка или истекает время ожидания, скрипт переходит к следующему товару.
   - Использование асинхронных функций и ограничение одновременных запросов обеспечивает баланс между скоростью и стабильностью работы скрипта.

---

### **Технологии и библиотеки**

- **Python 3.12**: Язык программирования, на котором написан скрипт.
- **asyncio**: Библиотека для написания асинхронного кода, позволяющая выполнять несколько операций ввода-вывода одновременно.
- **Playwright**: Инструмент для автоматизации браузера, поддерживающий асинхронную работу и позволяющий управлять браузером без интерфейса пользователя.
- **Playwright Stealth**: Модуль, который помогает скрыть автоматизированную природу браузера, чтобы избежать обнаружения сайтами и блокировки.
- **Polars**: Быстрая и эффективная библиотека для работы с данными, альтернативная pandas, оптимизированная для больших объемов данных.
- **ujson**: Быстрая реализация JSON-парсера и сериализатора, ускоряющая работу с JSON-файлами.
- **rapidfuzz**: Библиотека для быстрого и эффективного сравнения строк, используемая для объединения похожих характеристик.
- **aiofiles**: Библиотека для асинхронной работы с файлами, позволяющая не блокировать поток при чтении и записи файлов.

---

### **Пошаговый алгоритм работы скрипта**

1. **Инициализация**:

   - Настраивается логирование для отслеживания процесса выполнения скрипта.
   - Определяются глобальные переменные и списки URL сайтов для поиска.

2. **Чтение данных**:

   - Загружается файл `MTR.parquet` с помощью Polars.
   - Отфильтровываются записи, где поле `'Параметры'` пустое.

3. **Цикл обработки товаров**:

   - Для каждого товара из отфильтрованного списка:
     - Проверяется наличие выходного файла в папке `output`. Если файл уже существует, товар пропускается.
     - Извлекаются `'Наименование'`, `'Маркировка'` и `'код СКМТР'`.
     - Формируется поисковый запрос на основе длины полей `'Наименование'` и `'Маркировка'`.
     - Если поисковый запрос пустой, создается пустой JSON-файл, и скрипт переходит к следующему товару.

4. **Выполнение поиска и сбор ссылок**:

   - Инициализируется браузер с помощью Playwright и настраивается stealth-режим.
   - Выполняется поиск в Google с использованием поискового запроса и ограничением по сайтам.
   - Если результатов нет, выполняется повторный поиск без ограничения по сайтам.
   - Собираются ссылки на первые `n` результатов (по умолчанию 7).

5. **Обработка ссылок**:

   - Каждая ссылка обрабатывается асинхронно:
     - Переход по ссылке с учетом таймаутов.
     - Эмуляция пользовательского взаимодействия для обхода возможных защит на сайте.
     - Извлечение характеристик товара с помощью парсинга контента страницы.
     - Сохранение результатов во временный JSON-файл с именем, включающим `'код СКМТР'` и индекс ссылки.

6. **Объединение результатов**:

   - После обработки всех ссылок характеристики объединяются в один словарь.
   - Используется функция похожести строк из библиотеки rapidfuzz для объединения характеристик с похожими названиями.

7. **Сохранение окончательных данных**:

   - Объединенные характеристики сохраняются в файл JSON в папке `output`, имя файла соответствует `'код СКМТР'`.
   - Если характеристики не найдены, создается пустой JSON-файл.

8. **Очистка и завершение**:

   - Удаляются временные файлы, созданные при обработке ссылок.
   - Закрывается браузер и освобождаются ресурсы.

---

### **Обработка ошибок и надежность**

- **Таймауты**: Установлены таймауты на различные операции (переход по ссылке, ожидание загрузки страницы, ожидание элементов), чтобы избежать зависаний.
- **Обработка исключений**: Используются блоки `try-except` для перехвата и логирования ошибок без остановки работы всего скрипта.
- **Пустые файлы**: Если характеристики не найдены или произошла ошибка, создается пустой JSON-файл. Это предотвращает повторную обработку этого товара при следующем запуске скрипта.
- **Асинхронность**: Скрипт использует асинхронные функции для повышения скорости работы, но обработка товаров происходит последовательно для снижения нагрузки и повышения стабильности.

---

### **Преимущества использования скрипта**

- **Автоматизация**: Сокращает ручной труд по поиску и вводу параметров товаров.
- **Скорость**: Асинхронная обработка и эффективные библиотеки позволяют быстро обрабатывать большие объемы данных.
- **Точность**: Использование алгоритмов похожести строк повышает качество объединения характеристик из разных источников.
- **Масштабируемость**: Скрипт можно настроить на обработку любого количества товаров и адаптировать под другие источники данных или сайты.

---