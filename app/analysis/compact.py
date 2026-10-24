# app/analysis/compact.py
import json, math

def round_sig(x: float, sig: int = 4) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))

def _is_symmetric_matrix(d: dict) -> bool:
    keys = list(d)
    return (
        len(keys) > 1
        and all(isinstance(v, dict) and list(v) == keys for v in d.values())
        and all(d[a][b] == d[b][a] for a in keys for b in keys)
    )

def compact(obj, sig: int = 4):
    if isinstance(obj, float):
        return round_sig(obj, sig)
    if isinstance(obj, list):
        return [compact(v, sig) for v in obj]
    if isinstance(obj, dict):
        d = {k: compact(v, sig) for k, v in obj.items()}
        if _is_symmetric_matrix(d):  # 4x4 -> 6 unique pairs
            keys = list(d)
            return {f"{a}~{b}": d[a][b]
                    for i, a in enumerate(keys) for b in keys[i + 1:]}
        return d
    return obj

def fingerprint(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)