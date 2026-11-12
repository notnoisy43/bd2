# Що лишилось зробити (чекліст з інструкціями)

Порядок важливий: кроки 1–2 можна зробити будь-коли, кроки 3–8 ідуть один за одним.

---

## 1. Організаційне (Teams)

- [x] Заявити регіон у каналі Teams: `BoberRMDT: Benelux`
- [x] У тому ж треді написати міста й причину вибору C, наприклад:
  > A = Brussels, B = Antwerp, C = Amsterdam. Amsterdam is the largest and most expensive Benelux market, in another
  > country with its own neighbourhood system — the hardest realistic transfer test.

## 2. Імена команди та внесок кожного

Заглушки `_name_` / `_fill in_` треба замінити у трьох місцях:

- [x] `README.md`, розділ **Team contributions**
- [x] `scripts/build_presentation.py`: титульний слайд (`Team: <i>names</i>`) і таблиця на останньому слайді (`<i>name</i>` / `<i>fill in</i>`)
- [x] Перезібрати презентацію (потрібні локальні дані в `data/`, вони вже є):
  ```bash
  python scripts/build_presentation.py
  ```
  Результат: `docs/presentation.pdf`

---

## 3. Databricks Free Edition: логін і деплой ✅ (зроблено 12.11, воркспейс dbc-108bacd2-3110)

1. [ ] Зареєструватись / увійти на https://www.databricks.com/learn/free-edition і скопіювати URL воркспейсу
   (вигляду `https://dbc-xxxx.cloud.databricks.com`).
2. [ ] У терміналі в папці `D:\bd2`:
   ```bash
   databricks auth login --host https://<ваш-воркспейс>.cloud.databricks.com
   databricks bundle validate
   databricks bundle deploy
   ```
   - Якщо `validate` скаржиться на `warehouse_id`, значить склад у вас називається інакше, ніж "Serverless Starter Warehouse".
     Знайдіть назву в **SQL Warehouses** і виправте її в `databricks.yml` (`variables.warehouse_id.lookup.warehouse`).
   - Після деплою у воркспейсі з'являться два джоби (з префіксом `[dev <ваше ім'я>]`): `airbnb_city_etl` і `airbnb_lakehouse`,
     а також дашборд.

## 4. Запустити пайплайн ✅ (v1, v2, v3 пройшли на master; 4a пропущено, перший запуск C задокументовано локально + тег week1-final)

> Для максимальної чесності перший запуск міста C робиться з коду тижня 1. Нижче наведено обидва варіанти.

### 4a. Тиждень 1 + перший запуск Amsterdam рівно з замороженого коду (рекомендовано)

```bash
git checkout week1-final
databricks bundle deploy
databricks bundle run lakehouse                      # Brussels + Antwerp: ETL -> train v1 -> score -> report
databricks bundle run lakehouse --params 'cities=["amsterdam"]',retrain=false,model_version=v1
git checkout master
databricks bundle deploy
```
У коді week1-final параметр `cities` очікує JSON-список. Якщо CLI зламає лапки й коми (саме це баг P5), запустіть джоб
**в UI**: Jobs → airbnb_lakehouse → *Run now with different parameters* → `cities = ["amsterdam"]`, `retrain = false`.

### 4b. Після виправлень (поточний код)

```bash
databricks bundle run lakehouse --params cities=amsterdam,retrain=false,model_version=v1      # якщо пропустили 4a
databricks bundle run lakehouse --params cities=brussels+antwerp+amsterdam,train_cities=brussels+antwerp,model_version=v2
databricks bundle run lakehouse --params cities=brussels+antwerp+amsterdam,train_cities=brussels+antwerp+amsterdam,model_version=v3
```

**Якщо щось падає:**
- **Bronze не може завантажити файл** (serverless без інтернету): скачайте файли вручну й завантажте їх у
  Catalog → workspace → airbnb_benelux → Volumes → raw, за шляхом
  `/<місто>/<дата>/listings.csv.gz`, тобто `brussels/2026-09-20/`, `antwerp/2026-09-23/`, `amsterdam/2026-09-16/`.
  URL-адреси:
  - https://data.insideairbnb.com/belgium/bru/brussels/2026-09-20/data/listings.csv.gz
  - https://data.insideairbnb.com/belgium/vlg/antwerp/2026-09-23/data/listings.csv.gz
  - https://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2026-09-16/data/listings.csv.gz

  Потім перезапустіть джоб: bronze сам візьме файли з volume.
- **"price_model_vN.joblib exists"**: так задумано, бо версії моделей незмінні. Видаліть файл у volume `models` або візьміть нову назву версії.
- **Помилка імпорту `airbnb_lakehouse`**: перевірте, що бандл задеплоївся разом із папкою `src/` (Workspace → .bundle → … → files/src).
- **`for_each` / `{{tasks.plan.values.cities}}` не підтримується**: як запасний варіант запускайте `airbnb_city_etl` окремо для кожного міста
  (`--params city=amsterdam`), а потім ноутбуки 04/05/06 вручну з потрібними віджетами.

## 5. Перевірити цифри ✅ (Databricks: v1 32.47 / 38.53, v2 Amsterdam 109.03, v3 Amsterdam 83.58 — у межах різниці sklearn, тексти не міняли)

- [ ] SQL Editor:
  ```sql
  SELECT * FROM workspace.airbnb_benelux.gold_model_metrics ORDER BY model_version, city;
  SELECT * FROM workspace.airbnb_benelux.dq_metrics WHERE check_status = 'warn';
  ```
- [ ] Порівняти з локальними результатами: v1 Brussels 32.92 / Antwerp 39.01 / Amsterdam перший запуск 158.60;
  v2 Amsterdam 108.75; v3 Amsterdam 82.89.
  Невеликі розбіжності нормальні, бо на Databricks стоїть scikit-learn 1.6.1, а локально 1.8.
  **Якщо цифри відрізняються помітно**, оновіть їх у `README.md`, `docs/city_c/report.md`, заголовках дашборду
  (`scripts/build_dashboard.py` → `TITLES`, потім `python scripts/build_dashboard.py` і `databricks bundle deploy`) та в презентації.

## 6. Дашборд

- [x] Відкрити **Dashboards → "Benelux Airbnb: cities and price model"**. Якщо віджети порожні, натисніть *Refresh*.
- [x] Пройтись по 3 сторінках і перевірити, що кожен віджет рендериться. JSON дашборду я писав вручну, тому окремий віджет може
  не імпортуватись. У такому разі видаліть його й зберіть заново в редакторі: SQL кожного датасету лежить у `dashboards/queries.sql`.
- [ ] Якщо дашборд не імпортувався взагалі: **Dashboards → Create → Import** з файлу `dashboards/airbnb_benelux.lvdash.json`
  або зберіть його вручну з `queries.sql` (3 сторінки: City overview, Cross-city comparison, Model).
- [x] **Publish** дашборд, щоб його бачили інші.
- [ ] (Для презентації) зробити 2–3 скріншоти сторінок на випадок, якщо демо не запрацює.

## 7. Genie

- [x] Запустити один раз ноутбук `notebooks/07_uc_comments.sql` (виконано через SQL API, 37/37): він додає коментарі до таблиць і колонок, і Genie їх читає.
- [x] **Genie → New** (створено скриптом `scripts/build_genie_space.py --create`, space «Benelux Airbnb analyst»): склад "Serverless Starter Warehouse", далі пройти по розділах `genie/genie_space.md`:
  - Data: 5 gold-таблиць (розділ 1)
  - Instructions → Text: блок інструкцій (розділ 2)
  - Instructions → SQL queries: приклади SQL (розділ 3)
  - Sample questions (розділ 4)
- [x] Поставити три обов'язкові питання (усі 3 правильні, перевірено через API) й звірити з очікуваними відповідями:
  - "What is the median price of entire homes in City A?" → €138
  - "Which neighbourhoods in City B have the most listings?" → Historisch Centrum (397), Amandus - Atheneum (219), Theaterbuurt-Meir (194)
  - "Where does the model under-price listings the most?" → Oud-Oost, Amsterdam (v3, −€47), далі Centrum-Oost (−€38); лише A/B: Saint-Gilles, Brussels (−€12)
- [x] Якщо Genie відповідає неправильно, додати уточнення в інструкції й записати, що допомогло. Це потім піде на слайд
  "tricks we used to configure Genie" (розділ 5 у `genie_space.md`: там зараз обґрунтування, а не спостереження).

## 8. Репозиторій і здача

- [x] Створити порожній репозиторій на GitHub (private або public, як вимагає викладач).
- [x] Запушити разом із тегами (master + week1-final, week2-final) (теги — доказ, що перший запуск C був на коді тижня 1):
  ```bash
  git remote add origin https://github.com/<user>/<repo>.git
  git push -u origin --all
  git push origin --tags
  ```
  Папка `data/` у `.gitignore`, тому сирі дані не потраплять у репозиторій, і так має бути.
- [ ] Перевірити на GitHub, що відкриваються `README.md`, `docs/city_c/report.md`, `docs/presentation.pdf` і ноутбук з EDA.
- [ ] Team admin здає посилання на репозиторій.

## 9. Презентація (підготовка до виступу)

- [x] Розподілити слайди між учасниками (відповідно до внеску):
  - **Matvii Shumylovych**: 1 титул, 2 задача, 3 дані/EDA, 4 архітектура, 6 перший запуск Amsterdam, 10 демо Genie
  - **Dmytro Hrebenyuk**: 5 модель v1, 7 діагноз, 8 зміни (пайплайн vs модель), 9 цифри €172 → €83, 10 демо дашборду, 11 виклики
- [ ] Відрепетирувати демо: дашборд (3 сторінки) і Genie (3 питання з кроку 7).
- [ ] Головна історія виступу: *перший запуск на Amsterdam пройшов без жодної помилки, і саме це було проблемою* → діагноз →
  виправлення (пайплайн vs модель) → €172 → €109 → €83, а решта розриву пропорційна (MAPE 26 % vs 23 %).

---

### Де що лежить (шпаргалка)

| що | файл |
|---|---|
| головний звіт про місто C | `docs/city_c/report.md` |
| логи першого запуску (докази) | `docs/city_c/*.log`, `first_run_metrics.csv` |
| EDA + огляд витоків | `notebooks/00_eda_brussels_antwerp.ipynb` |
| архітектура | `docs/architecture.md` |
| картки моделей | `docs/model_v1.md`, `docs/model_v3.md` |
| джоби | `databricks.yml`, `resources/jobs.yml`, `resources/dashboard.yml` |
| дашборд | `dashboards/airbnb_benelux.lvdash.json` (+ `queries.sql`) |
| Genie | `genie/genie_space.md`, `notebooks/07_uc_comments.sql` |
| презентація | `docs/presentation.pdf` (джерело: `scripts/build_presentation.py`) |
| локальний запуск | `scripts/run_local.py` (команди в README) |
