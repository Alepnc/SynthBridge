import numpy as np
import pandas as pd

NA_TOKEN = "__NA__"


def generate(data: pd.DataFrame, n_rows: int, params: dict, info: dict) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    seed = int(info["seed"])
    max_training_rows = int(params["max_training_rows"])
    max_trees = int(params["max_trees"])
    max_condition_cols = int(params["max_condition_cols"])
    max_depth = int(params["max_depth"])
    min_samples_leaf = int(params["min_samples_leaf"])
    low_card = int(params["low_cardinality_threshold"])
    shuffle_cols = params["column_order"] == "random"
    reproduce_missing = bool(params["reproduce_missing"])

    if len(data) == 0:
        raise ValueError("Il dataset di input è vuoto.")

    original_columns = list(data.columns)

    cat_cols = set(info["categorical_columns"])
    for col in info["numeric_columns"]:
        s = pd.to_numeric(data[col], errors="coerce").dropna()
        if len(s) and s.nunique() <= low_card and np.allclose(s, np.round(s)):
            cat_cols.add(col)
    num_cols = [c for c in original_columns if c not in cat_cols]

    work = pd.DataFrame(index=data.index)
    for col in original_columns:
        if col in cat_cols:
            work[col] = data[col].astype("object").where(data[col].notna(), NA_TOKEN).astype(str)
        else:
            s = pd.to_numeric(data[col], errors="coerce").astype(float)
            work[col] = s.fillna(s.median() if s.notna().any() else 0.0)

    na_mask = pd.DataFrame(
        {c: pd.to_numeric(data[c], errors="coerce").isna() for c in num_cols}, index=data.index
    )
    na_rate = {c: float(na_mask[c].mean()) for c in num_cols}

    if len(work) > max_training_rows:
        idx = work.sample(n=max_training_rows, random_state=seed).index
    else:
        idx = work.index
    train = work.loc[idx].reset_index(drop=True)
    train_na = na_mask.loc[idx].reset_index(drop=True)

    order = list(original_columns)
    if shuffle_cols:
        perm = np.random.default_rng(seed).permutation(len(order))
        order = [original_columns[i] for i in perm]
    print(f"Ordine di generazione: {order}")

    syn = pd.DataFrame(index=range(n_rows))
    first = order[0]
    pool = train[first]
    if first in num_cols and (~train_na[first]).any():
        pool = pool[~train_na[first]]
    syn[first] = pool.sample(n=n_rows, replace=True, random_state=seed).reset_index(drop=True)

    n_steps = max(len(order) - 1, 1)
    for step, target in enumerate(order[1:], start=1):
        info["progress"](0.05 + 0.9 * step / n_steps, f"colonna {target}")
        rng = np.random.default_rng(seed + step)

        feats = order[max(0, step - max_condition_cols):step]
        X_train = pd.get_dummies(train[feats], drop_first=False)
        X_gen = pd.get_dummies(syn[feats], drop_first=False)
        X_gen = X_gen.reindex(columns=X_train.columns, fill_value=0)

        if target in cat_cols:
            model = RandomForestClassifier(
                n_estimators=max_trees, random_state=seed + step, n_jobs=-1,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            )
            model.fit(X_train, train[target])
            proba = model.predict_proba(X_gen)
            cum = proba.cumsum(axis=1)
            u = rng.random(n_rows)[:, None]
            idx = np.minimum((u > cum).sum(axis=1), len(model.classes_) - 1)
            syn[target] = model.classes_[idx]
        else:
            model = RandomForestRegressor(
                n_estimators=max_trees, random_state=seed + step, n_jobs=-1,
                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            )
            obs = ~train_na[target].to_numpy()
            if obs.sum() < 2:
                obs = np.ones(len(train), dtype=bool)
            model.fit(X_train[obs], train[target][obs])
            pred = model.predict(X_gen)
            residuals = train[target].to_numpy()[obs] - model.predict(X_train[obs])
            syn[target] = pred + rng.choice(residuals, size=n_rows, replace=True)

    out = pd.DataFrame(index=range(n_rows))
    for col in original_columns:
        orig = data[col]
        s = syn[col]
        if col in cat_cols:
            s = s.astype(str).where(s.astype(str) != NA_TOKEN, np.nan)
            if pd.api.types.is_bool_dtype(orig.dtype) or str(orig.dtype) == "boolean":
                s = s.map({"True": True, "False": False}).astype("boolean")
            elif pd.api.types.is_integer_dtype(orig.dtype):
                s = pd.to_numeric(s, errors="coerce").astype("Int64")
            elif isinstance(orig.dtype, pd.CategoricalDtype):
                s = pd.Categorical(s, categories=orig.cat.categories)
            elif pd.api.types.is_float_dtype(orig.dtype):
                s = pd.to_numeric(s, errors="coerce")
        else:
            s = s.astype(float)
            if reproduce_missing and na_rate.get(col, 0) > 0:
                rng_na = np.random.default_rng(seed + 10_000 + original_columns.index(col))
                s = s.mask(rng_na.random(n_rows) < na_rate[col])
            if pd.api.types.is_integer_dtype(orig.dtype):
                s = s.round().astype("Int64")
        out[col] = s

    info["progress"](1.0, "fine")
    return out[original_columns]
