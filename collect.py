"""
collect.py — camada de comparação entre execuções.

Divisão de responsabilidades
----------------------------
runs/<nome>.json   um JSON enxuto por execução: config, encoding, exact_solution
                   e analysis. Sem cost_history (vai para o .npz), o que leva o
                   arquivo de ~400 kB para ~15 kB.
arrays/<nome>.npz  históricos pesados e kets.
summary.csv        UMA LINHA POR EXECUÇÃO, com todos os escalares. É aqui que se
                   compara, não no JSON.
all_results.json   artefato consolidado, gerado DEPOIS por consolidate(), para
                   acompanhar a tese e o repositório.

Escrever um JSON único durante a execução seria frágil: uma queda no meio da
escrita corrompe tudo o que já foi rodado, duas execuções paralelas disputam o
mesmo arquivo, e reexecutar uma configuração vira leitura-modificação-escrita do
conjunto inteiro. Por isso a consolidação acontece no fim, a partir dos arquivos
por execução, que continuam sendo a fonte da verdade.
"""
from pathlib import Path
from typing import Dict, List, Optional
import json


def load_summary(folder) -> "object":
    """Carrega summary.csv como DataFrame (ou lista de dicts, sem pandas)."""
    csv_path = Path(folder) / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"summary.csv não encontrado em {folder}")
    try:
        import pandas as pd
        return pd.read_csv(csv_path)
    except ImportError:
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as fh:
            return list(_csv.DictReader(fh))


def load_run(folder, name: str) -> Dict:
    """Carrega o JSON de uma execução pelo nome do experimento."""
    p = Path(folder) / "runs" / f"{name}.json"
    if not p.exists():
        p = Path(folder) / "runs" / f"{name}_results.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_arrays(folder, name: str):
    """Carrega os históricos pesados (.npz) de uma execução."""
    import numpy as np
    p = Path(folder) / "arrays" / f"{name}_state.npz"
    return np.load(p, allow_pickle=True)


def iter_runs(folder):
    """Itera sobre os JSONs por execução, em ordem de nome."""
    for p in sorted((Path(folder) / "runs").glob("*.json")):
        yield p.stem, json.loads(p.read_text(encoding="utf-8"))


def consolidate(folder, out: str = "all_results.json",
                keep: Optional[List[str]] = None) -> Path:
    """Junta os JSONs por execução num único arquivo para publicação.

    Mantém só os blocos de análise: a ~15 kB por execução, 200 execuções dão
    ~3 MB, manejável. Rode isto no FIM da varredura, nunca durante.
    """
    keep = keep or ["config", "encoding", "exact_solution", "analysis",
                    "status", "timing_seconds", "timestamps"]
    folder = Path(folder)
    merged = {"schema_version": "consolidated-1.0", "source_folder": str(folder),
              "runs": {}}
    for name, d in iter_runs(folder):
        merged["runs"][d.get("experiment_name", name)] = {
            k: d[k] for k in keep if k in d}
    merged["n_runs"] = len(merged["runs"])
    out_path = folder / out
    out_path.write_text(json.dumps(merged, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    return out_path


def compare(folder, by: str, metric: str = "rho", sweep: Optional[str] = None):
    """Agrega uma métrica por uma coluna, com média e desvio sobre as sementes.

    Uso típico das varreduras combinadas:
        compare(f, by="positions_R")                 # escopo (varredura C)
        compare(f, by="lambda_disc", sweep="B")      # calibração
        compare(f, by="initial_state", sweep="D")    # GKP x vácuo
    """
    df = load_summary(folder)
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        raise RuntimeError("compare() requer pandas; use load_summary() sem ele.")
    if sweep is not None and "sweep" in df.columns:
        df = df[df["sweep"] == sweep]
    valid = df[df.get("state_norm_final", 1.0) >= 0.99] if "state_norm_final" in df else df
    dropped = len(df) - len(valid)
    if dropped:
        print(f"[aviso] {dropped} execuções descartadas por state_norm < 0,99")
    return valid.groupby(by)[metric].agg(["count", "mean", "std"])
