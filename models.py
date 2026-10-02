import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11

RANDOM_STATE = 42


def prepare_dataframe(path="data/cardekho.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден.")

    df = pd.read_csv(path)
    df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")
    df["brand"] = df["name"].astype(str).str.split().str[0]
    top_brands = df["brand"].value_counts().nlargest(10).index
    df["brand"] = df["brand"].where(df["brand"].isin(top_brands), "Other")
    df["car_age"] = int(df["year"].max()) - df["year"]
    df = df.drop(columns=["name", "year"])
    df = df[df["owner"] != "Test Drive Car"].copy()
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    df = df.rename(columns={"mileage(km/ltr/kg)": "mileage"})
    df_encoded = pd.get_dummies(
        df,
        columns=["fuel", "seller_type", "transmission", "owner", "brand"],
        drop_first=True,
        dtype=int,
    )
    return df_encoded


def evaluate_models(models, X_tr, X_te, y_tr, y_te, space_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}
    preds = {}

    print(f"=== Модели: {space_name} ===")
    for name, model in models.items():
        cv_r2 = cross_val_score(model, X_tr, y_tr, cv=kf, scoring="r2")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2 = float(r2_score(y_te, y_pred))
        mape = float(mean_absolute_percentage_error(y_te, y_pred) * 100)

        fitted[name] = model
        preds[name] = y_pred
        rows.append(
            {
                "Модель": name,
                "Пространство": space_name,
                "R2_CV_mean": cv_r2.mean(),
                "R2_CV_std": cv_r2.std(),
                "R2_Test": r2,
                "RMSE_Test": rmse,
                "MAPE_Test_%": mape,
            }
        )
        print(
            f"{name:<28} | R2(CV)={cv_r2.mean():.4f}±{cv_r2.std():.4f} | "
            f"R2(Test)={r2:.4f} | RMSE={rmse:,.0f} | MAPE={mape:.2f}%"
        )

    return pd.DataFrame(rows), fitted, preds


def main():
    show_plots = "--no-show" not in sys.argv

    df_encoded = prepare_dataframe()
    X = df_encoded.drop(columns=["selling_price"])
    y = df_encoded["selling_price"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Обучающая выборка: {X_train.shape[0]}")
    print(f"Тестовая выборка:  {X_test.shape[0]}")
    print(f"Число признаков:   {X_train.shape[1]}")

    models_orig = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=10000),
    }
    results_orig, fitted_orig, preds_orig = evaluate_models(
        models_orig,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        space_name=f"Исходные признаки ({len(feature_names)})",
    )

    pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_scaled)
    eigenvalues = pca_full.explained_variance_
    exp_var_ratio = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var_ratio)

    k_kaiser = int((eigenvalues > 1.0).sum())
    k_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    k_opt = k_kaiser if k_kaiser >= 2 else k_90
    if cum_var[k_opt - 1] < 0.85:
        k_opt = k_90

    print(f"\nКомпонент с lambda > 1: {k_kaiser}")
    print(f"Компонент для >=90% дисперсии: {k_90}")
    print(f"Выбрано k = {k_opt} (накопленная дисперсия = {cum_var[k_opt - 1] * 100:.2f}%)")

    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", lw=2, markersize=5)
        axes[0].axhline(1.0, color="red", ls="--", lw=2, label="Критерий Кайзера (lambda = 1)")
        axes[0].axvline(k_opt, color="green", ls=":", lw=2, label=f"Выбрано k = {k_opt}")
        axes[0].set_title("График каменистой осыпи (Scree Plot)")
        axes[0].set_xlabel("Номер главной компоненты")
        axes[0].set_ylabel("Собственное значение")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(range(1, len(cum_var) + 1), cum_var, "ro-", lw=2, markersize=5)
        axes[1].axhline(0.90, color="orange", ls="--", label="Порог 90%")
        axes[1].axvline(k_opt, color="green", ls=":", lw=2, label=f"Выбрано k = {k_opt}")
        axes[1].set_title("Кумулятивная объяснённая дисперсия")
        axes[1].set_xlabel("Число главных компонент")
        axes[1].set_ylabel("Доля объяснённой дисперсии")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close()

    pca = PCA(n_components=k_opt, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    models_pca = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=100.0, random_state=RANDOM_STATE, max_iter=10000),
    }
    results_pca, fitted_pca, preds_pca = evaluate_models(
        models_pca,
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        space_name=f"Главные компоненты ({k_opt})",
    )

    summary = pd.concat([results_orig, results_pca], ignore_index=True)
    summary_sorted = summary.sort_values("R2_Test", ascending=False).reset_index(drop=True)
    print("\nИтоговая сравнительная таблица:")
    print(summary_sorted.round(4).to_string(index=False))

    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.barplot(data=summary, x="Модель", y="R2_Test", hue="Пространство", ax=axes[0], edgecolor="black")
        axes[0].set_title("Сравнение R2 (Test)")
        sns.barplot(data=summary, x="Модель", y="RMSE_Test", hue="Пространство", ax=axes[1], edgecolor="black")
        axes[1].set_title("Сравнение RMSE (Test)")
        sns.barplot(data=summary, x="Модель", y="MAPE_Test_%", hue="Пространство", ax=axes[2], edgecolor="black")
        axes[2].set_title("Сравнение MAPE (Test), %")
        plt.tight_layout()
        plt.show()
        plt.close()

    best = summary_sorted.iloc[0]
    print("\nЛучшая модель по R2 на тесте:")
    print(f"  {best['Модель']} | {best['Пространство']}")
    print(
        f"  R2={best['R2_Test']:.4f}, RMSE={best['RMSE_Test']:,.0f}, "
        f"MAPE={best['MAPE_Test_%']:.2f}%"
    )

    weights_df = pd.DataFrame(
        {
            "Признак": feature_names,
            "Linear": fitted_orig["Linear"].coef_,
            "Ridge": fitted_orig["Ridge"].coef_,
            "Lasso": fitted_orig["Lasso"].coef_,
        }
    )
    weights_df.to_csv("weights.csv", index=False)

    payload = {
        "dataset": "data/cardekho.csv",
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "pca_components": int(k_opt),
        "pca_explained_variance": float(cum_var[k_opt - 1]),
        "results_original": results_orig.to_dict(orient="records"),
        "results_pca": results_pca.to_dict(orient="records"),
    }
    with open("laba1_model_weights.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nСохранено: weights.csv, laba1_model_weights.json")


if __name__ == "__main__":
    main()
