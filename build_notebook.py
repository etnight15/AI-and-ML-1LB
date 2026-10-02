import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))

                             
md(r"""
# Лабораторная работа 1. Линейная регрессия и факторный анализ

| | |
|---|---|
| **Выполнил** | **Марков Данил** |
| **Группа** | **АВТ-314** |
| **Проверил** | **Антонянц Егор Николаевич** |

**Датасет:** [Car Price Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset)  
**Файл данных:** `data/cardekho.csv`  
**Целевая переменная:** `selling_price` (цена продажи автомобиля)
""")

                                       
md(r"""
## 1. Введение

**Цель работы** — изучить построение моделей линейной регрессии на реальных данных, оценить влияние мультиколлинеарности на качество прогноза и сравнить модели, обученные на исходных признаках и на главных компонентах (PCA).

**Задачи работы:**
1. Загрузить датасет о ценах автомобилей и провести первичный анализ.
2. Визуализировать распределения признаков и целевой переменной.
3. Выполнить предобработку: удаление пропусков, кодирование категориальных признаков, стандартизация.
4. Построить матрицу корреляций и рассчитать коэффициенты VIF для диагностики мультиколлинеарности.
5. Обучить модели линейной, гребневой (Ridge) и лассо (Lasso) регрессии с разбиением train/test и кросс-валидацией; оценить качество по RMSE, R² и MAPE.
6. Устранить мультиколлинеарность и снизить размерность методом главных компонент (PCA) после стандартизации.
7. Повторить обучение тех же моделей на главных компонентах и сравнить метрики.

Целевая переменная — непрерывная (`selling_price`), поэтому используется регрессия, а не классификация.
""")

                                  
md(r"""
## 2. Описание датасета и первичный анализ

Ниже подключаются необходимые библиотеки: `pandas`/`numpy` для работы с таблицами, `matplotlib`/`seaborn` для графиков, `scikit-learn` для моделей и PCA, `statsmodels` для расчёта VIF.
""")

code(r"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
""")

md(r"""
Библиотеки загружены. Зафиксирован `RANDOM_STATE = 42`, чтобы разбиение выборки и результаты кросс-валидации воспроизводились при повторном запуске ноутбука.
""")

                                 
md(r"""
### 2.1. Загрузка данных
""")

code(r"""
df_raw = pd.read_csv('data/cardekho.csv')

print(f'Размерность исходного датасета: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов')
print('\nСписок столбцов:')
print(list(df_raw.columns))
print('\nТипы данных:')
print(df_raw.dtypes)
print('\nПервые 5 строк:')
display(df_raw.head())
""")

md(r"""
Датасет содержит **8128** наблюдений и **12** признаков. Источник — площадка CarDekho (Индия).

**Кратко о признаках:**
- `name` — модель автомобиля (текстовый идентификатор);
- `year` — год выпуска;
- `selling_price` — **целевая переменная**, цена продажи;
- `km_driven` — пробег, км;
- `fuel` — тип топлива (Diesel, Petrol, CNG, LPG);
- `seller_type` — тип продавца;
- `transmission` — коробка передач (Manual / Automatic);
- `owner` — номер владельца;
- `mileage(km/ltr/kg)` — расход/пробег на единицу топлива;
- `engine` — объём двигателя, см³;
- `max_power` — максимальная мощность (в файле хранится как текст);
- `seats` — число посадочных мест.

Далее выполняется первичный анализ пропусков и описательных статистик.
""")

                           
md(r"""
### 2.2. Пропуски, дубликаты и описательная статистика
""")

code(r"""
print('Пропущенные значения по столбцам:')
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Пропуски': missing, 'Доля, %': missing_pct})
display(missing_df[missing_df['Пропуски'] > 0])

dup_count = df_raw.duplicated().sum()
print(f'\nЧисло полных дубликатов: {dup_count}')

print('\nОписательная статистика числовых столбцов:')
display(df_raw.describe().T.round(2))

print('\nРаспределение категориальных признаков:')
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    counts = df_raw[col].value_counts()
    pct = (df_raw[col].value_counts(normalize=True) * 100).round(2)
    print(f'\n{col}:')
    display(pd.DataFrame({'Количество': counts, 'Доля, %': pct}))
""")

md(r"""
**Выводы по первичному анализу:**
- Пропуски сосредоточены в `mileage`, `engine`, `max_power`, `seats` (около 2–3% строк). Их доля невелика, поэтому в предобработке такие строки будут удалены.
- Есть полные дубликаты — их тоже удалим, чтобы не искажать обучение.
- Цена продажи сильно варьируется (от десятков тысяч до миллионов) — распределение, скорее всего, асимметричное.
- Среди продавцов преобладают частные лица (`Individual`), среди коробок передач — механика (`Manual`).
""")

                                     
md(r"""
### 2.3. Визуализация распределений признаков и целевой переменной
""")

code(r"""
df_viz = df_raw.copy()
df_viz['max_power_num'] = pd.to_numeric(df_viz['max_power'], errors='coerce')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.histplot(df_viz['selling_price'], bins=40, kde=True, ax=axes[0, 0], color='#2b5c8f', edgecolor='black')
axes[0, 0].axvline(df_viz['selling_price'].mean(), color='red', ls='--', label=f"Среднее: {df_viz['selling_price'].mean():,.0f}")
axes[0, 0].axvline(df_viz['selling_price'].median(), color='green', ls='-', label=f"Медиана: {df_viz['selling_price'].median():,.0f}")
axes[0, 0].set_title('Распределение цены продажи')
axes[0, 0].set_xlabel('selling_price')
axes[0, 0].legend(fontsize=8)

sns.histplot(df_viz['year'], bins=25, kde=False, ax=axes[0, 1], color='#1f77b4', edgecolor='black')
axes[0, 1].set_title('Распределение года выпуска')
axes[0, 1].set_xlabel('year')

sns.histplot(df_viz['km_driven'], bins=40, kde=True, ax=axes[0, 2], color='#2ca02c', edgecolor='black')
axes[0, 2].set_title('Распределение пробега')
axes[0, 2].set_xlabel('km_driven')

sns.histplot(df_viz['mileage(km/ltr/kg)'].dropna(), bins=35, kde=True, ax=axes[1, 0], color='#ff7f0e', edgecolor='black')
axes[1, 0].set_title('Распределение расхода/пробега топлива')
axes[1, 0].set_xlabel('mileage')

sns.histplot(df_viz['engine'].dropna(), bins=35, kde=True, ax=axes[1, 1], color='#9467bd', edgecolor='black')
axes[1, 1].set_title('Распределение объёма двигателя')
axes[1, 1].set_xlabel('engine')

sns.histplot(df_viz['max_power_num'].dropna(), bins=35, kde=True, ax=axes[1, 2], color='#8c564b', edgecolor='black')
axes[1, 2].set_title('Распределение мощности')
axes[1, 2].set_xlabel('max_power')

plt.tight_layout()
plt.show()
""")

md(r"""
Графики показывают, что цена продажи и пробег имеют **правостороннюю асимметрию**: большинство автомобилей в среднем ценовом сегменте, но есть дорогие выбросы. Год выпуска смещён к более новым машинам (2010–2020). Объём двигателя и мощность также асимметричны — это типично для рынка подержанных авто.
""")

code(r"""
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

sns.boxplot(data=df_viz, x='fuel', y='selling_price', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Цена в зависимости от типа топлива')
axes[0, 0].tick_params(axis='x', rotation=15)

sns.boxplot(data=df_viz, x='transmission', y='selling_price', ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('Цена в зависимости от коробки передач')

sns.boxplot(data=df_viz, x='owner', y='selling_price', ax=axes[1, 0], palette='Blues')
axes[1, 0].set_title('Цена в зависимости от числа владельцев')
axes[1, 0].tick_params(axis='x', rotation=20)

sns.scatterplot(data=df_viz.sample(n=min(2000, len(df_viz)), random_state=RANDOM_STATE),
                x='year', y='selling_price', hue='transmission', alpha=0.5, ax=axes[1, 1])
axes[1, 1].set_title('Связь года выпуска и цены')

plt.tight_layout()
plt.show()
""")

md(r"""
Видно, что **автоматическая коробка** обычно дороже механики, а более новые автомобили стоят существенно выше. Число предыдущих владельцев также влияет на цену: у первого владельца медианная цена выше. Эти зависимости подтверждают целесообразность включения категориальных признаков в регрессионные модели.
""")

                                        
md(r"""
## 3. Подготовка данных

### 3.1. Очистка, преобразование типов и кодирование
""")

code(r"""
df = df_raw.copy()

df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

df['brand'] = df['name'].astype(str).str.split().str[0]
top_brands = df['brand'].value_counts().nlargest(10).index
df['brand'] = df['brand'].where(df['brand'].isin(top_brands), 'Other')

max_year = int(df['year'].max())
df['car_age'] = max_year - df['year']

df = df.drop(columns=['name', 'year'])

df = df[df['owner'] != 'Test Drive Car'].copy()

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f'Удалено дубликатов: {before - len(df)}')

before_na = len(df)
df = df.dropna().reset_index(drop=True)
print(f'Удалено строк с пропусками: {before_na - len(df)}')
print(f'Итоговый размер после очистки: {df.shape}')

df = df.rename(columns={'mileage(km/ltr/kg)': 'mileage'})

print('\nПервые строки после очистки:')
display(df.head())
""")

md(r"""
**Что сделано на этом шаге:**
1. `max_power` преобразован в числовой тип (`pd.to_numeric`).
2. Из `name` извлечён бренд; оставлены 10 самых частых, остальные — `Other` (снижает размерность one-hot кодирования).
3. Вместо года выпуска введён `car_age` — возраст авто.
4. Удалены дубликаты, пропуски и редкий класс `Test Drive Car`.
5. Столбец `name` убран: это почти уникальный идентификатор, а не устойчивый признак для линейной модели.
""")

code(r"""
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

print(f'Размерность после One-Hot Encoding: {df_encoded.shape}')
print('Список признаков:')
print(list(df_encoded.columns))
display(df_encoded.head())
""")

md(r"""
Категориальные признаки закодированы методом **One-Hot Encoding** с параметром `drop_first=True`. Это создаёт бинарные столбцы и убирает одну категорию-эталон, чтобы не вводить идеальную мультиколлинеарность между дамми-переменными. Целевая переменная `selling_price` остаётся числовой.
""")

                                   
md(r"""
### 3.2. Матрица корреляций и анализ связи с целевой переменной
""")

code(r"""
num_for_corr = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'car_age']
corr_full = df_encoded.corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_full.loc[num_for_corr, num_for_corr], annot=True, fmt='.2f',
            cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Корреляционная матрица основных числовых признаков')
plt.tight_layout()
plt.show()

target_corr = corr_full['selling_price'].drop('selling_price').sort_values(key=abs, ascending=False)
print('Корреляция признаков с selling_price (по модулю, топ-15):')
display(target_corr.head(15).to_frame('corr_with_target').round(3))
""")

md(r"""
**Интерпретация корреляций:**
- С ценой сильнее всего связаны мощность (`max_power`), объём двигателя (`engine`), возраст (`car_age`, обычно отрицательная связь) и тип трансмиссии.
- Между `engine` и `max_power` ожидается высокая положительная корреляция — это первый сигнал возможной **мультиколлинеарности**.
- `mileage` и `car_age`/`km_driven` также могут быть связаны между собой.

Для формальной проверки мультиколлинеарности далее рассчитывается VIF.
""")

                                         
code(r"""
plt.figure(figsize=(14, 11))
sns.heatmap(corr_full, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.2, cbar_kws={'shrink': 0.7})
plt.title('Полная корреляционная матрица после кодирования')
plt.tight_layout()
plt.show()
""")

md(r"""
Полная матрица подтверждает наличие блоков взаимосвязанных признаков (технические характеристики двигателя и часть категориальных дамми). Это обосновывает применение PCA на следующем этапе.
""")

                           
md(r"""
### 3.3. Диагностика мультиколлинеарности: VIF
""")

code(r"""
X_all = df_encoded.drop(columns=['selling_price'])
y_all = df_encoded['selling_price']

scaler_vif = StandardScaler()
X_scaled_vif = pd.DataFrame(scaler_vif.fit_transform(X_all), columns=X_all.columns)

vif_df = pd.DataFrame({
    'Признак': X_scaled_vif.columns,
    'VIF': [variance_inflation_factor(X_scaled_vif.values, i) for i in range(X_scaled_vif.shape[1])]
}).sort_values('VIF', ascending=False).reset_index(drop=True)

print('Коэффициенты вздутия дисперсии (VIF):')
display(vif_df.round(3))

plt.figure(figsize=(10, max(5, 0.28 * len(vif_df))))
bars = plt.barh(vif_df['Признак'], vif_df['VIF'], color='#3182bd', edgecolor='black')
plt.axvline(5, color='orange', ls='--', lw=1.5, label='Порог VIF = 5')
plt.axvline(10, color='red', ls='--', lw=1.5, label='Порог VIF = 10')
plt.xlabel('VIF')
plt.title('Оценка мультиколлинеарности (VIF)')
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.show()

print(f"Признаков с VIF > 5: {(vif_df['VIF'] > 5).sum()}")
print(f"Признаков с VIF > 10: {(vif_df['VIF'] > 10).sum()}")
""")

md(r"""
**Как читать VIF:**
- VIF ≈ 1 — мультиколлинеарности почти нет;
- VIF > 5 — умеренная/заметная мультиколлинеарность;
- VIF > 10 — сильная мультиколлинеарность, оценки коэффициентов OLS нестабильны.

Если часть признаков превышает пороги 5–10, линейная модель на исходных признаках может давать неустойчивые веса. Регуляризация (Ridge/Lasso) и PCA — естественные способы смягчить эту проблему.
""")

                                  
md(r"""
### 3.4. Разделение на train/test и стандартизация
""")

code(r"""
X = df_encoded.drop(columns=['selling_price'])
y = df_encoded['selling_price']
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

print(f'Обучающая выборка: {X_train.shape[0]} объектов')
print(f'Тестовая выборка:  {X_test.shape[0]} объектов')
print(f'Число признаков:   {X_train.shape[1]}')
print('\nПроверка: среднее ≈ 0, стд ≈ 1 на train после стандартизации:')
display(pd.DataFrame({'mean': X_train_scaled.mean().round(4), 'std': X_train_scaled.std().round(4)}).head(8))
""")

md(r"""
Выборка разделена в пропорции **80/20**. Стандартизация (`StandardScaler`) обучается **только на train**, а к test применяется тот же преобразователь — это исключает утечку данных. Стандартизация обязательна перед Ridge/Lasso и PCA, потому что эти методы чувствительны к масштабу признаков.
""")

                                            
md(r"""
## 4. Ход работы

### 4.1. Регрессионные модели на исходных (стандартизированных) признаках

Обучаются три модели:
- **LinearRegression (OLS)** — обычная линейная регрессия;
- **Ridge** — L2-регуляризация, уменьшает веса коррелирующих признаков;
- **Lasso** — L1-регуляризация, может обнулять незначимые признаки.

Для каждой модели выполняется 5-fold кросс-валидация по R² на train и расчёт RMSE, R², MAPE на test.
""")

code(r"""
def evaluate_models(models, X_tr, X_te, y_tr, y_te, space_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    preds = {}

    print(f'=== Модели: {space_name} ===')
    for name, model in models.items():
        cv_r2 = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2 = float(r2_score(y_te, y_pred))
        mape = float(mean_absolute_percentage_error(y_te, y_pred) * 100)

        fitted[name] = model
        preds[name] = y_pred
        rows.append({
            'Модель': name,
            'Пространство': space_name,
            'R² (CV mean)': cv_r2.mean(),
            'R² (CV std)': cv_r2.std(),
            'R² (Test)': r2,
            'RMSE (Test)': rmse,
            'MAPE (Test), %': mape
        })
        print(f"{name:<28} | R²(CV)={cv_r2.mean():.4f}±{cv_r2.std():.4f} | "
              f"R²(Test)={r2:.4f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}%")

    return pd.DataFrame(rows), fitted, preds

models_orig = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=10.0, random_state=RANDOM_STATE),
    'Lasso': Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=10000)
}

results_orig, fitted_orig, preds_orig = evaluate_models(
    models_orig, X_train_scaled, X_test_scaled, y_train, y_test,
    space_name=f'Исходные признаки ({len(feature_names)})'
)

display(results_orig.round(4))
""")

md(r"""
Метрики показывают, насколько хорошо модели объясняют цену на отложенной выборке:
- **R²** — доля объяснённой дисперсии (чем ближе к 1, тем лучше);
- **RMSE** — типичная ошибка в единицах цены (чем меньше, тем лучше);
- **MAPE** — средняя относительная ошибка в процентах.

Ridge и Lasso обычно ближе к OLS, если регуляризация умеренная; при сильной мультиколлинеарности регуляризация стабилизирует решение. Ниже — визуальное сравнение метрик и график «факт vs прогноз» для линейной модели.
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
plot_df = results_orig.copy()

sns.barplot(data=plot_df, x='Модель', y='R² (Test)', ax=axes[0], palette='Blues_r', edgecolor='black')
axes[0].set_title('R² на тесте (исходные признаки)')
axes[0].set_ylim(0, max(0.05, plot_df['R² (Test)'].max() * 1.15))

sns.barplot(data=plot_df, x='Модель', y='RMSE (Test)', ax=axes[1], palette='Reds_r', edgecolor='black')
axes[1].set_title('RMSE на тесте (исходные признаки)')

sns.barplot(data=plot_df, x='Модель', y='MAPE (Test), %', ax=axes[2], palette='Greens_r', edgecolor='black')
axes[2].set_title('MAPE на тесте (исходные признаки)')

plt.tight_layout()
plt.show()

y_hat = preds_orig['Linear']
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test, y_hat, alpha=0.35, edgecolor='none', color='#2b5c8f')
lims = [min(y_test.min(), y_hat.min()), max(y_test.max(), y_hat.max())]
ax.plot(lims, lims, 'r--', lw=2, label='Идеальный прогноз')
ax.set_xlabel('Фактическая цена')
ax.set_ylabel('Прогноз')
ax.set_title('LinearRegression: факт vs прогноз (исходные признаки)')
ax.legend()
plt.tight_layout()
plt.show()
""")

md(r"""
Столбчатые диаграммы удобны для сравнения моделей «на одном экране». Диаграмма рассеяния «факт–прогноз» показывает, где модель ошибается сильнее всего: обычно это дорогие автомобили-выбросы, которые плохо описываются линейной зависимостью.
""")

code(r"""
coef_df = pd.DataFrame({
    'Признак': feature_names,
    'Linear': fitted_orig['Linear'].coef_,
    'Ridge': fitted_orig['Ridge'].coef_,
    'Lasso': fitted_orig['Lasso'].coef_
}).sort_values('Linear', key=abs, ascending=False)

print('Топ-12 признаков по модулю коэффициента LinearRegression:')
display(coef_df.head(12).round(2))

print(f"Intercept Linear: {fitted_orig['Linear'].intercept_:,.2f}")
print(f"Число нулевых коэффициентов Lasso: {(np.abs(fitted_orig['Lasso'].coef_) < 1e-8).sum()} из {len(feature_names)}")
""")

md(r"""
Таблица коэффициентов помогает интерпретировать модель: положительный вес увеличивает прогноз цены, отрицательный — уменьшает. У Lasso часть коэффициентов может обнулиться — это встроенный отбор признаков. При мультиколлинеарности знаки и величины коэффициентов OLS нужно трактовать осторожно.
""")

                           
md(r"""
### 4.2. Устранение мультиколлинеарности методом главных компонент (PCA)

Перед PCA данные уже стандартизированы. Строится полный PCA, анализируется график каменистой осыпи (scree plot) и накопленная доля объяснённой дисперсии. Число компонент выбирается по критерию Кайзера (λ > 1) и/или порогу объяснённой дисперсии ≈ 85–95%.
""")

code(r"""
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_scaled)
eigenvalues = pca_full.explained_variance_
exp_var_ratio = pca_full.explained_variance_ratio_
cum_var = np.cumsum(exp_var_ratio)

pca_info = pd.DataFrame({
    'Компонента': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Собств. значение λ': eigenvalues,
    'Доля дисперсии': exp_var_ratio,
    'Накопл. дисперсия': cum_var
})
display(pca_info.round(4))

k_kaiser = int((eigenvalues > 1.0).sum())
k_90 = int(np.searchsorted(cum_var, 0.90) + 1)
k_opt = k_kaiser if k_kaiser >= 2 else k_90
if cum_var[k_opt - 1] < 0.85:
    k_opt = k_90

print(f'Компонент с λ > 1 (критерий Кайзера): {k_kaiser}')
print(f'Компонент для ≥90% дисперсии: {k_90}')
print(f'Выбрано компонент k = {k_opt} (накопленная дисперсия = {cum_var[k_opt-1]*100:.2f}%)')
""")

md(r"""
Таблица собственных значений показывает, сколько дисперсии несёт каждая главная компонента. Далее строится **график каменистой осыпи**: по оси X — номер компоненты, по оси Y — собственное значение. «Локоть» графика и линия λ = 1 помогают выбрать разумное число компонент.
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', lw=2, markersize=5)
axes[0].axhline(1.0, color='red', ls='--', lw=2, label='Критерий Кайзера (λ = 1)')
axes[0].axvline(k_opt, color='green', ls=':', lw=2, label=f'Выбрано k = {k_opt}')
axes[0].set_title('График каменистой осыпи (Scree Plot)')
axes[0].set_xlabel('Номер главной компоненты')
axes[0].set_ylabel('Собственное значение λ')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(cum_var) + 1), cum_var, 'ro-', lw=2, markersize=5, label='Накопленная дисперсия')
axes[1].axhline(0.90, color='orange', ls='--', label='Порог 90%')
axes[1].axvline(k_opt, color='green', ls=':', lw=2, label=f'Выбрано k = {k_opt}')
axes[1].set_title('Кумулятивная объяснённая дисперсия')
axes[1].set_xlabel('Число главных компонент')
axes[1].set_ylabel('Доля объяснённой дисперсии')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

md(r"""
Scree plot и кумулятивная кривая обосновывают выбор числа компонент `k`. После выбора PCA переобучается с `n_components=k`, и исходные признаки заменяются некоррелированными главными компонентами — это как раз способ снизить мультиколлинеарность и размерность.
""")

code(r"""
pca = PCA(n_components=k_opt, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
pc_names = [f'PC{i+1}' for i in range(k_opt)]

loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=pc_names)

plt.figure(figsize=(min(14, 1.1 * k_opt + 4), 8))
sns.heatmap(loadings.iloc[:, :min(10, k_opt)], annot=False, cmap='coolwarm', center=0, linewidths=0.3)
plt.title(f'Факторные нагрузки (первые {min(10, k_opt)} компонент)')
plt.ylabel('Исходные признаки')
plt.xlabel('Главные компоненты')
plt.tight_layout()
plt.show()

print(f'Размерность train до PCA:  {X_train_scaled.shape}')
print(f'Размерность train после PCA: {X_train_pca.shape}')
print('Корреляции между первыми 5 PC на train (должны быть ≈ 0):')
display(pd.DataFrame(X_train_pca[:, :min(5, k_opt)], columns=pc_names[:min(5, k_opt)]).corr().round(4))
""")

md(r"""
Тепловая карта нагрузок показывает, какие исходные признаки сильнее всего входят в каждую компоненту. Корреляции между самими PC близки к нулю — мультиколлинеарность в пространстве компонент устранена по построению метода.
""")

                                        
md(r"""
### 4.3. Регрессионные модели на главных компонентах
""")

code(r"""
models_pca = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=10.0, random_state=RANDOM_STATE),
    'Lasso': Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=10000)
}

results_pca, fitted_pca, preds_pca = evaluate_models(
    models_pca, X_train_pca, X_test_pca, y_train, y_test,
    space_name=f'Главные компоненты ({k_opt})'
)

display(results_pca.round(4))
""")

md(r"""
На главных компонентах те же три алгоритма обучаются заново. Если исходные признаки были сильно коррелированы, метрики после PCA могут стать стабильнее; если же отброшено слишком много информативной дисперсии, качество может немного упасть. Сравнение «до/после» — ключевой результат работы.
""")

                                  
md(r"""
### 4.4. Сравнение метрик до и после PCA
""")

code(r"""
summary = pd.concat([results_orig, results_pca], ignore_index=True)
summary_sorted = summary.sort_values('R² (Test)', ascending=False).reset_index(drop=True)

print('Итоговая сравнительная таблица:')
display(summary_sorted.round(4))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
summary['Метка'] = summary['Модель'] + '\n' + summary['Пространство'].str.replace('Исходные признаки', 'Orig').str.replace('Главные компоненты', 'PCA')

sns.barplot(data=summary, x='Модель', y='R² (Test)', hue='Пространство', ax=axes[0], edgecolor='black')
axes[0].set_title('Сравнение R² (Test)')
axes[0].legend(fontsize=8)

sns.barplot(data=summary, x='Модель', y='RMSE (Test)', hue='Пространство', ax=axes[1], edgecolor='black')
axes[1].set_title('Сравнение RMSE (Test)')
axes[1].legend(fontsize=8)

sns.barplot(data=summary, x='Модель', y='MAPE (Test), %', hue='Пространство', ax=axes[2], edgecolor='black')
axes[2].set_title('Сравнение MAPE (Test), %')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.show()

best = summary_sorted.iloc[0]
print('\nЛучшая модель по R² на тесте:')
print(f"  {best['Модель']} | {best['Пространство']}")
print(f"  R²={best['R² (Test)']:.4f}, RMSE={best['RMSE (Test)']:,.0f}, MAPE={best['MAPE (Test), %']:.2f}%")
""")

md(r"""
**Как интерпретировать сравнение:**
- Если после PCA R² почти не упал (или вырос), а RMSE/MAPE не ухудшились — сжатие признакового пространства удалось без большой потери информации.
- Если качество заметно просело, значит отброшенные компоненты содержали полезный сигнал; тогда стоит увеличить `k` или оставить регуляризованные модели на исходных признаках.
- Ridge/Lasso на исходных данных часто конкурируют с PCA+Linear: оба подхода борются с мультиколлинеарностью разными механизмами.
""")

                                     
md(r"""
## 5. Заключение

В работе на датасете цен автомобилей CarDekho выполнены все этапы пайплайна: загрузка и EDA, очистка и кодирование, диагностика мультиколлинеарности (корреляции + VIF), обучение Linear/Ridge/Lasso, снижение размерности PCA и повторное моделирование.

**Основные выводы:**
1. Цена продажи связана с мощностью, объёмом двигателя, возрастом автомобиля, типом трансмиссии и другими характеристиками — линейные модели улавливают существенную часть этой зависимости.
2. Между техническими признаками присутствует мультиколлинеарность (высокие корреляции и повышенный VIF), что делает оценки коэффициентов OLS менее устойчивыми.
3. Ridge и Lasso стабилизируют решение за счёт регуляризации; Lasso дополнительно выполняет отбор признаков.
4. PCA после стандартизации устраняет корреляции между предикторами и уменьшает число признаков; итоговые метрики позволяют выбрать компромисс между сжатием пространства и точностью прогноза.
5. Возможные пути улучшения: подбор гиперпараметров `alpha` через GridSearchCV, логарифмирование целевой переменной из-за асимметрии цены, нелинейные модели (деревья/градиентный бустинг) при сохранении интерпретируемого baseline на линейных моделях.

Практический результат лабораторной работы — воспроизводимое сравнение качества регрессии **до и после** факторного сжатия признаков.

Полный программный код вынесен в отдельные файлы `laba1_analysis.py` (загрузка, EDA, корреляции, VIF) и `laba1_models.py` (регрессия, PCA, сравнение метрик), по аналогии с примерами лабораторных работ.
""")

                                     
md(r"""
## 6. Список источников

1. Dataset: Sukhmandeep Singh Brar. *Car Price Prediction Dataset*. Kaggle.  
   https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset
2. Pedregosa et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011.  
   https://scikit-learn.org/stable/
3. James G., Witten D., Hastie T., Tibshirani R. *An Introduction to Statistical Learning*. Springer.
4. Документация pandas: https://pandas.pydata.org/docs/
5. Документация statsmodels (VIF): https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
6. Jolliffe I.T. *Principal Component Analysis*. Springer.
""")

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'pygments_lexer': 'ipython3'
    }
}

out_path = 'Lab1_Linear_Regression_Factor_Analysis.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook saved: {out_path}, cells: {len(cells)}')
