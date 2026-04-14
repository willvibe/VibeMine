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
        else:
            top_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            col_info["top_value"] = str(top_val) if top_val is not None else None

        is_id = col.lower() in ('id', 'idx', 'no', 'number', 'serial', 'uid', 'key') or col.endswith('_id')
        is_low_variance = col_info["unique_count"] <= 1
        col_info["is_id"] = is_id
        col_info["is_low_variance"] = is_low_variance

        columns.append(col_info)

    preview = df.head(5).fillna("NaN").to_dict(orient="records")

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            value_counts = df[col].value_counts()
            if len(value_counts) >= 2:
                ratio = value_counts.iloc[0] / value_counts.iloc[-1] if value_counts.iloc[-1] > 0 else 0
                if ratio > 5:
                    imbalance_detected = True
                    imbalance_ratio = max(imbalance_ratio, ratio)

    return {
        "shape": shape,
        "columns": columns,
        "column_names": list(df.columns),
        "preview": preview,
        "imbalance_detected": imbalance_detected,
        "imbalance_ratio": round(imbalance_ratio, 2),
    }