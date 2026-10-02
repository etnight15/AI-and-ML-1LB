import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

RANDOM_STATE = 42


def load_and_clean(path="data/cardekho.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден.")

    df_raw = pd.read_csv(path)
    print(f"Размерность исходного датасета: {df_raw.shape[0]} строк, {df_raw.shape[1]} столбцов")
    print("\nПропущенные значения по столбцам:")
    print(df_raw.isnull().sum().to_string())
    print(f"\nЧисло полных дубликатов: {df_raw.duplicated().sum()}")

    df = df_raw.copy()
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")
    df["brand"] = df["name"].astype(str).str.split().str[0]
    top_brands = df["brand"].value_counts().nlargest(10).index
    df["brand"] = df["brand"].where(df["brand"].isin(top_brands), "Other")
    max_year = int(df["year"].max())
    df["car_age"] = max_year - df["year"]
    df = df.drop(columns=["name", "year"])
    df = df[df["owner"] != "Test Drive Car"].copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nУдалено дубликатов: {before - len(df)}")
    before_na = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Удалено строк с пропусками: {before_na - len(df)}")
    df = df.rename(columns={"mileage(km/ltr/kg)": "mileage"})
    print(f"Итоговый размер после очистки: {df.shape}")
    return df_raw, df


def plot_distributions(df_raw, show_plots=True):
    df_viz = df_raw.copy()
    df_viz["max_power_num"] = pd.to_numeric(df_viz["max_power"], errors="coerce")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    sns.histplot(df_viz["selling_price"], bins=40, kde=True, ax=axes[0, 0], color="#2b5c8f", edgecolor="black")
    axes[0, 0].set_title("Распределение цены продажи")
    sns.histplot(df_viz["year"], bins=25, kde=False, ax=axes[0, 1], color="#1f77b4", edgecolor="black")
    axes[0, 1].set_title("Распределение года выпуска")
    sns.histplot(df_viz["km_driven"], bins=40, kde=True, ax=axes[0, 2], color="#2ca02c", edgecolor="black")
    axes[0, 2].set_title("Распределение пробега")
    sns.histplot(df_viz["mileage(km/ltr/kg)"].dropna(), bins=35, kde=True, ax=axes[1, 0], color="#ff7f0e", edgecolor="black")
    axes[1, 0].set_title("Распределение mileage")
    sns.histplot(df_viz["engine"].dropna(), bins=35, kde=True, ax=axes[1, 1], color="#9467bd", edgecolor="black")
    axes[1, 1].set_title("Распределение engine")
    sns.histplot(df_viz["max_power_num"].dropna(), bins=35, kde=True, ax=axes[1, 2], color="#8c564b", edgecolor="black")
    axes[1, 2].set_title("Распределение max_power")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    sns.boxplot(data=df_viz, x="fuel", y="selling_price", hue="fuel", ax=axes[0, 0], palette="Set2", legend=False)
    axes[0, 0].set_title("Цена и тип топлива")
    axes[0, 0].tick_params(axis="x", rotation=15)
    sns.boxplot(data=df_viz, x="transmission", y="selling_price", hue="transmission", ax=axes[0, 1], palette="pastel", legend=False)
    axes[0, 1].set_title("Цена и коробка передач")
    sns.boxplot(data=df_viz, x="owner", y="selling_price", hue="owner", ax=axes[1, 0], palette="Blues", legend=False)
    axes[1, 0].set_title("Цена и число владельцев")
    axes[1, 0].tick_params(axis="x", rotation=20)
    sns.scatterplot(
        data=df_viz.sample(n=min(2000, len(df_viz)), random_state=RANDOM_STATE),
        x="year",
        y="selling_price",
        hue="transmission",
        alpha=0.5,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Год выпуска и цена")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()


def encode_features(df):
    cat_cols = ["fuel", "seller_type", "transmission", "owner", "brand"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    print(f"Размерность после One-Hot Encoding: {df_encoded.shape}")
    return df_encoded


def correlation_and_vif(df_encoded, show_plots=True):
    num_for_corr = ["selling_price", "km_driven", "mileage", "engine", "max_power", "seats", "car_age"]
    corr_full = df_encoded.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_full.loc[num_for_corr, num_for_corr],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )
    plt.title("Корреляционная матрица основных числовых признаков")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    target_corr = corr_full["selling_price"].drop("selling_price").sort_values(key=abs, ascending=False)
    print("\nКорреляция признаков с selling_price (топ-15 по модулю):")
    print(target_corr.head(15).round(3).to_string())

    X_all = df_encoded.drop(columns=["selling_price"])
    scaler_vif = StandardScaler()
    X_scaled_vif = pd.DataFrame(scaler_vif.fit_transform(X_all), columns=X_all.columns)
    vif_df = pd.DataFrame(
        {
            "Признак": X_scaled_vif.columns,
            "VIF": [variance_inflation_factor(X_scaled_vif.values, i) for i in range(X_scaled_vif.shape[1])],
        }
    ).sort_values("VIF", ascending=False).reset_index(drop=True)

    print("\nКоэффициенты VIF:")
    print(vif_df.round(3).to_string(index=False))
    print(f"\nПризнаков с VIF > 5: {(vif_df['VIF'] > 5).sum()}")
    print(f"Признаков с VIF > 10: {(vif_df['VIF'] > 10).sum()}")

    plt.figure(figsize=(10, max(5, 0.28 * len(vif_df))))
    plt.barh(vif_df["Признак"], vif_df["VIF"], color="#3182bd", edgecolor="black")
    plt.axvline(5, color="orange", ls="--", lw=1.5, label="Порог VIF = 5")
    plt.axvline(10, color="red", ls="--", lw=1.5, label="Порог VIF = 10")
    plt.xlabel("VIF")
    plt.title("Оценка мультиколлинеарности (VIF)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.close()

    return corr_full, vif_df


def main():
    show_plots = "--no-show" not in sys.argv
    df_raw, df = load_and_clean()
    print("\nОписательная статистика числовых столбцов:")
    print(df.describe().T.round(2).to_string())
    plot_distributions(df_raw, show_plots=show_plots)
    df_encoded = encode_features(df)
    correlation_and_vif(df_encoded, show_plots=show_plots)
    df_encoded.to_csv("data/cardekho_encoded.csv", index=False)
    print("\nСохранено: data/cardekho_encoded.csv")


if __name__ == "__main__":
    main()
