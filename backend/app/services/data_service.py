import os
import uuid
import pandas as pd
from app.config import UPLOAD_DIR


def save_upload_file(file_bytes: bytes, filename: str) -> str:
    unique_id = uuid.uuid4().hex[:8]
    safe_name = f"{unique_id}_{filename}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)
    with open(filepath, "wb") as f:
        f.write(file_bytes)
    return safe_name


def parse_csv(filename: str) -> pd.DataFrame:
    filepath = os.path.join(UPLOAD_DIR, filename)
    df = pd.read_csv(filepath)
    return df


def get_data_profile(df: pd.DataFrame) -> dict:
    shape = list(df.shape)
    columns = []
    imbalance_detected = False
    imbalance_ratio = 0.0

    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "missing_ratio": round(float(df[col].isnull().mean()), 4),
            "unique_count": int(df[col].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["mean"] = round(float(df[col].mean()), 4) if not df[col].isnull().all() else None
            col_info["std"] = round(float(df[col].std()), 4) if not df[col].isnull().all() else None
            col_info["min"] = round(float(df[col].min()), 4) if not df[col].isnull().all() else None
            col_info["max"] = round(float(df[col].max()), 4) if not df[col].isnull().all() else None
            col_info["p25"] = round(float(df[col].quantile(0.25)), 4) if not df[col].isnull().all() else None
            col_info["p50"] = round(float(df[col].quantile(0.50)), 4) if not df[col].isnull().all() else None
            col_info["p75"] = round(float(df[col].quantile(0.75)), 4) if not df[col].isnull().all() else None
        else:
            top_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            col_info["top_value"] = str(top_val) if top_val is not None else None
            val_counts = df[col].value_counts()
            col_info["value_counts"] = {str(k): int(v) for k, v in val_counts.head(10).items()}

        is_id = col.lower() in ('id', 'idx', 'no', 'number', 'serial', 'uid', 'key') or col.endswith('_id')
        is_low_variance = col_info["unique_count"] <= 1
        col_info["is_id"] = is_id
        col_info["is_low_variance"] = is_low_variance

        columns.append(col_info)

    preview = df.head(5).fillna("NaN").to_dict(orient="records")

    # Calculate class distribution for each potential target column
    class_distributions = {}
    for col in df.columns:
        # Check object columns or numeric columns with <= 10 unique values (potential classification targets)
        if df[col].dtype == 'object' or (pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10 and df[col].nunique() >= 2):
            value_counts = df[col].value_counts()
            if len(value_counts) >= 2:
                ratio = value_counts.iloc[0] / value_counts.iloc[-1] if value_counts.iloc[-1] > 0 else 0
                if ratio > 2:
                    imbalance_detected = True
                    imbalance_ratio = max(imbalance_ratio, ratio)
                    class_distributions[col] = {
                        str(k): int(v) for k, v in value_counts.items()
                    }

    describe_data = df.describe(include='all').fillna("N/A").to_dict()

    info_list = []
    total_rows = len(df)
    for col in df.columns:
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_vals = df[col].nunique()
        dtype_str = str(df[col].dtype)
        if dtype_str == 'object':
            sample_val = df[col].dropna().iloc[0] if non_null > 0 else ''
            info_list.append({
                "column": col,
                "dtype": dtype_str,
                "non_null": int(non_null),
                "null": int(null_count),
                "unique": int(unique_vals),
                "sample": str(sample_val)[:50] if sample_val else '',
            })
        else:
            info_list.append({
                "column": col,
                "dtype": dtype_str,
                "non_null": int(non_null),
                "null": int(null_count),
                "unique": int(unique_vals),
            })

    return {
        "shape": shape,
        "columns": columns,
        "column_names": list(df.columns),
        "preview": preview,
        "imbalance_detected": imbalance_detected,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "class_distributions": class_distributions,
        "describe": describe_data,
        "info": info_list,
        "total_rows": total_rows,
    }