# api.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import json
from typing import Dict, Any, List
from io import StringIO

from eda_cli.eda import summarize_dataset, missing_table, compute_quality_flags
from eda_cli.visualizations import plot_missingness_heatmap

app = FastAPI(title="EDA Service", version="1.0.0")


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "service": "eda-api"}


@app.post("/quality")
async def quality_check(data: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Проверка качества данных из JSON."""
    try:
        df = pd.DataFrame(data)
        flags = compute_quality_flags(df)
        return {"flags": flags}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/quality-from-csv")
async def quality_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Проверка качества данных из CSV-файла."""
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        flags = compute_quality_flags(df)
        return {"flags": flags}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/quality-flags-from-csv")
async def quality_flags_from_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Полный набор флагов качества из CSV-файла.
    Включает все эвристики, реализованные в HW03.
    """
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        
        summary = summarize_dataset(df)
        missing = missing_table(df)
        flags = compute_quality_flags(df)
        
        return {
            "flags": flags,
            "summary": summary,
            "missing_statistics": missing
        }
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Файл CSV пуст")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Ошибка парсинга CSV")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Ошибка кодировки файла")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")


@app.post("/summary-from-csv")
async def summary_from_csv(
    file: UploadFile = File(...),
    include_missing: bool = True,
    include_stats: bool = True,
    include_flags: bool = True
) -> Dict[str, Any]:
    """
    Полная JSON-сводка по данным из CSV-файла.
    Аналог CLI-режима --json-summary из HW03.
    """
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        
        result = {
            "dataset_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        }
        
        if include_stats:
            result["statistics"] = summarize_dataset(df)
        
        if include_missing:
            result["missing_data"] = missing_table(df)
        
        if include_flags:
            result["quality_flags"] = compute_quality_flags(df)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/head")
async def get_head(file: UploadFile = File(...), n: int = 5) -> Dict[str, Any]:
    """
    Возвращает первые n строк из CSV-файла.
    Аналог CLI-команды head.
    """
    try:
        if n <= 0:
            raise HTTPException(status_code=400, detail="Параметр n должен быть положительным числом")
        
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        
        head_df = df.head(n)
        
        return {
            "count": len(head_df),
            "total_rows": len(df),
            "requested_n": n,
            "data": head_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/sample")
async def get_sample(
    file: UploadFile = File(...),
    n: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Возвращает случайную выборку из CSV-файла.
    Аналог CLI-команды sample.
    """
    try:
        if n <= 0:
            raise HTTPException(status_code=400, detail="Параметр n должен быть положительным числом")
        
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        
        if n > len(df):
            n = len(df)
        
        sample_df = df.sample(n=n, random_state=random_state)
        
        return {
            "count": len(sample_df),
            "total_rows": len(df),
            "requested_n": n,
            "random_state": random_state,
            "data": sample_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/missingness-heatmap")
async def missingness_heatmap(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Генерирует данные для тепловой карты пропущенных значений.
    Возвращает структурированные данные для построения графика.
    """
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")
        df = pd.read_csv(StringIO(content_str))
        
        heatmap_data = plot_missingness_heatmap(df, return_data=True)
        
        return {
            "heatmap_data": heatmap_data,
            "shape": df.shape,
            "missing_percentage": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 