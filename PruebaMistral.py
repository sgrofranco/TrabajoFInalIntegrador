# benchmark_mistral_npc.py
# Simulación de 5 NPCs llamando a Mistral con intervalos independientes (~20s),
# registro de latencia, validación JSON, tokens (desde resp.usage si está),
# agregación por minuto y p50/p95 + costo/min opcional.
#
# Uso:
#   export MISTRAL_API_KEY=...
#   # (opcional) export PRICE_IN_PER_1K=0.15 ; export PRICE_OUT_PER_1K=0.60
#   python benchmark_mistral_npc.py --prompt_json jsonformatter.json --model ministral-8b-2410 --minutes 20 --plots
#
import os, json, time, math, argparse, asyncio, random
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional
from matplotlib import pyplot as plt
import pandas as pd
from mistralai import Mistral

def now_s() -> float: return time.perf_counter()

def approx_token_count(text: str) -> int:
    if not text: return 0
    by_chars = max(1, len(text) // 4)
    by_words = int(len(text.split()) * 1.3)
    return min(by_chars, by_words)

def minute_bucket(t0: float, t: float, bucket_sec: int) -> int:
    return int((t - t0) // bucket_sec)

def p_quantile(values: List[float], q: float) -> float:
    if not values: return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi]-s[lo])*(pos-lo)

def extract_json_object(text: str) -> Optional[dict]:
    """Intenta extraer un objeto JSON {accion, dialogo} de un string."""
    if not text: return None
    # Limpieza básica por si vienen backticks o texto extra
    txt = text.strip()
    # Intento directo
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "accion" in data and "dialogo" in data:
            return data
    except Exception:
        pass
    # Heurística: recortar desde el primer '{' hasta el último '}'
    try:
        a, b = txt.find("{"), txt.rfind("}")
        if 0 <= a < b:
            data = json.loads(txt[a:b+1])
            if isinstance(data, dict) and "accion" in data and "dialogo" in data:
                return data
    except Exception:
        pass
    return None

@dataclass
class Record:
    npc_id: int
    t_send: float
    t_recv: float
    latency_s: Optional[float]
    ok_json: int
    tokens_in: int
    tokens_out: int
    error: Optional[str]
    minute: int

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_json", required=True, help="Ruta al JSON con 'messages' y opcionalmente 'model'.")
    p.add_argument("--model", default=None, help="Modelo a usar (p.ej. ministral-8b-2410). Prioriza sobre el JSON.")
    p.add_argument("--npcs", type=int, default=5)
    p.add_argument("--minutes", type=int, default=1)
    p.add_argument("--mean_interval", type=float, default=20.0, help="Media de intervalo (seg) por NPC.")
    p.add_argument("--bucket_sec", type=int, default=60, help="Bucket de agregación (seg).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plots", action="store_true")
    return p.parse_args()

async def main():
    args = build_args()

    API_KEY = "slybhMEGJnm8XCMIFOrS943grlep5T1u"
    if not API_KEY:
        raise SystemExit("Falta MISTRAL_API_KEY")

    PRICE_IN_PER_1K = 0.0001 #float(os.environ.get("PRICE_IN_PER_1K", "0"))
    PRICE_OUT_PER_1K = 0.0001 #float(os.environ.get("PRICE_OUT_PER_1K", "0"))

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        base_payload = json.load(f)
    MESSAGES = base_payload.get("messages")
    if not isinstance(MESSAGES, list):
        raise SystemExit("El JSON debe contener 'messages' (lista).")
    MODEL = args.model or base_payload.get("model") or "mistral-large-latest"

    client = Mistral(api_key=API_KEY)

    results: List[Record] = []
    t0 = now_s()
    end_time = t0 + args.minutes * 60

    async def npc_loop(npc_id: int, seed: int):
        rng = random.Random(seed)
        def sample_interval():  # Exponencial(media = mean_interval)
            lam = 1.0 / max(0.01, args.mean_interval)
            return rng.expovariate(lam)

        while now_s() < end_time:
            payload_messages = deepcopy(MESSAGES)

            for m in payload_messages:
                if isinstance(m.get("content"), (dict, list)):
                    m["content"] = json.dumps(m["content"], ensure_ascii=False, indent=2)

            est_tokens_in = approx_token_count(json.dumps(payload_messages, ensure_ascii=False))
            t_send = now_s()
            tokens_in, tokens_out = est_tokens_in, 0
            latency, ok_flag, err = None, 0, None

            try:
                # mistralai es sync; lo mandamos a thread para no bloquear
                resp = await asyncio.to_thread(client.chat.complete, model=MODEL, messages=payload_messages)
                t_recv = now_s()
                latency = t_recv - t_send

                content = resp.choices[0].message.content if (resp and resp.choices) else ""
                data = extract_json_object(content)
                ok_flag = 1 if data is not None else 0

                # Uso real de tokens si viene en resp.usage
                try:
                    usage = getattr(resp, "usage", None)
                    if usage:
                        # El SDK de Mistral suele exponerlo como atributos o dict; cubrimos ambos
                        prompt_tokens = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens", None)
                        completion_tokens = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens", None)
                        total_tokens = getattr(usage, "total_tokens", None) or usage.get("total_tokens", None)
                        if isinstance(prompt_tokens, int): tokens_in = prompt_tokens
                        if isinstance(completion_tokens, int): tokens_out = completion_tokens
                        # Si solo viene total_tokens, aproximamos reparto conservador
                        if not isinstance(prompt_tokens, int) and isinstance(total_tokens, int):
                            tokens_in = max(0, int(total_tokens * 0.7))
                            tokens_out = total_tokens - tokens_in
                    else:
                        tokens_out = approx_token_count(content)
                except Exception:
                    tokens_out = approx_token_count(content)

                results.append(Record(
                    npc_id=npc_id, t_send=t_send, t_recv=t_recv,
                    latency_s=latency, ok_json=ok_flag,
                    tokens_in=tokens_in, tokens_out=tokens_out,
                    error=None, minute=minute_bucket(t0, t_recv, args.bucket_sec)
                ))

            except Exception as e:
                t_recv = now_s()
                results.append(Record(
                    npc_id=npc_id, t_send=t_send, t_recv=t_recv,
                    latency_s=None, ok_json=0, tokens_in=est_tokens_in, tokens_out=0,
                    error=str(e), minute=minute_bucket(t0, t_recv, args.bucket_sec)
                ))
                await asyncio.sleep(min(10.0, 1.0 + random.random()*2.0))
            else:
                await asyncio.sleep(sample_interval())

    tasks = [asyncio.create_task(npc_loop(i, args.seed + i + 1)) for i in range(args.npcs)]
    await asyncio.gather(*tasks)

    # DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        print("No se registraron resultados."); return

    df.to_csv("metrics_raw.csv", index=False, encoding="utf-8")
    df_ok = df[df["latency_s"].notna()].copy()

    # p50/p95 globales
    latencies = df_ok["latency_s"].tolist()
    p50, p95 = p_quantile(latencies, 0.50), p_quantile(latencies, 0.95)

    # Agregado por minuto (sum 5 NPCs)
    df_ok["tokens_total"] = df_ok["tokens_in"] + df_ok["tokens_out"]
    agg = df_ok.groupby("minute").agg(
        tokens_in_min=("tokens_in", "sum"),
        tokens_out_min=("tokens_out", "sum"),
        tokens_total_min=("tokens_total", "sum"),
        latency_mean_s=("latency_s", "mean"),
        latency_med_s=("latency_s", "median"),
        requests=("latency_s", "count")
    ).reset_index()

    # Errores e inválidos por minuto
    errs = df[df["error"].notna()].groupby("minute").size().rename("errors").reset_index()
    invalid = df_ok[df_ok["ok_json"] == 0].groupby("minute").size().rename("invalid_json").reset_index()
    agg = agg.merge(errs, on="minute", how="left").merge(invalid, on="minute", how="left")
    agg["errors"] = agg["errors"].fillna(0).astype(int)
    agg["invalid_json"] = agg["invalid_json"].fillna(0).astype(int)

    # Costo por minuto (si hay precios)
    if PRICE_IN_PER_1K > 0 or PRICE_OUT_PER_1K > 0:
        agg["cost_min_usd"] = (agg["tokens_in_min"]/1000.0)*PRICE_IN_PER_1K + (agg["tokens_out_min"]/1000.0)*PRICE_OUT_PER_1K
    else:
        agg["cost_min_usd"] = 0.0

    agg.to_csv("agg_minute.csv", index=False, encoding="utf-8")

    # Resumen
    total_reqs = len(df)
    total_ok = df_ok.shape[0]
    total_err = int(df["error"].notna().sum())
    total_invalid = int((df_ok["ok_json"] == 0).sum())

    print("\n=== RESUMEN SESIÓN ===")
    print(f"Modelo: {MODEL}")
    print(f"NPCs: {args.npcs} | Duración: {args.minutes} min | Media intervalo: {args.mean_interval:.1f}s")
    print(f"Requests totales: {total_reqs} | OK: {total_ok} | Errores: {total_err} | JSON inválido: {total_invalid}")
    print(f"Latencia p50: {p50:.3f}s | p95: {p95:.3f}s")
    print(f"Tokens totales (in+out): {int(df_ok['tokens_total'].sum())}")
    if PRICE_IN_PER_1K > 0 or PRICE_OUT_PER_1K > 0:
        print(f"Costo total estimado: ${agg['cost_min_usd'].sum():.4f} USD")
    else:
        print("Costo: N/A (sin precios configurados).")

    if args.plots:
        import matplotlib.pyplot as plt
        # Tokens/min
        plt.figure()
        plt.plot(agg["minute"], agg["tokens_total_min"])
        plt.title(f"Tokens/min (sum 5 NPCs) — {MODEL}")
        plt.xlabel("minuto"); plt.ylabel("tokens/min"); plt.grid(True, alpha=0.3)
        plt.savefig("plot_tokens_per_min.png", dpi=150, bbox_inches="tight")

        # Latencia/min + p50/p95 global
        plt.figure()
        plt.plot(agg["minute"], agg["latency_med_s"], label="mediana/min")
        plt.plot(agg["minute"], agg["latency_mean_s"], label="media/min")
        plt.axhline(p50, linestyle="--", label=f"p50 global = {p50:.3f}s")
        plt.axhline(p95, linestyle="--", label=f"p95 global = {p95:.3f}s")
        plt.title(f"Latencia por minuto — {MODEL}")
        plt.xlabel("minuto"); plt.ylabel("segundos")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig("plot_latency_per_min.png", dpi=150, bbox_inches="tight")

        # Costo/min
        plt.figure()
        plt.plot(agg["minute"], agg["cost_min_usd"])
        plt.title(f"Costo/min — {MODEL}")
        plt.xlabel("minuto"); plt.ylabel("USD/min"); plt.grid(True, alpha=0.3)
        plt.savefig("plot_cost_per_min.png", dpi=150, bbox_inches="tight")
        
        # ----------------------------
        # (B) NUEVOS: GRÁFICOS POR REQUEST (barras finas)
        # ----------------------------
        # Ordenar por momento de recepción
        df_ok_sorted = df_ok.sort_values("t_recv").reset_index(drop=True).copy()
        df_ok_sorted["request_idx"] = df_ok_sorted.index

        # (B1) Latencia por request (una barra por request)
        plt.figure(figsize=(12, 4))
        plt.bar(df_ok_sorted["request_idx"], df_ok_sorted["latency_s"], width=0.8)
        plt.title(f"Latencia por request — {MODEL}")
        plt.xlabel("request # (orden temporal)")
        plt.ylabel("segundos")
        plt.grid(True, axis="y", alpha=0.3)

        # Líneas de referencia p50/p95 globales con etiquetas
        plt.axhline(p50, linestyle="--", color="tab:orange", label=f"p50 global = {p50:.3f}s")
        plt.axhline(p95, linestyle="--", color="tab:red", label=f"p95 global = {p95:.3f}s")
        plt.legend()

        plt.savefig("plot_latency_per_request.png", dpi=150, bbox_inches="tight")


        # (B2) Tokens por request (apilados in/out)
        plt.figure(figsize=(10, 4))
        # barras apiladas: primero IN, luego OUT encima
        plt.bar(df_ok_sorted["request_idx"], df_ok_sorted["tokens_in"], width=0.8)
        plt.bar(df_ok_sorted["request_idx"], df_ok_sorted["tokens_out"], width=0.8,
                bottom=df_ok_sorted["tokens_in"])
        plt.title(f"Tokens por request (in + out) — {MODEL}")
        plt.xlabel("request # (orden temporal)")
        plt.ylabel("tokens")
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig("plot_tokens_per_request.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    asyncio.run(main())
