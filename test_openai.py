#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, time, math, argparse, asyncio, random, csv, statistics
from typing import Any, Dict, List, Tuple

# --- Matplotlib: backend "Agg" para guardar sin abrir ventanas ---
import matplotlib
def _use_backend(show: bool):
    if not show:
        matplotlib.use("Agg")
_use_backend(False)
import matplotlib.pyplot as plt

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("Falta openai. Instalá con: pip install openai>=1.0.0")

# -------------------- Helpers --------------------
def to_responses_input(data: Any) -> List[Dict[str, str]]:
    """
    Acepta {"messages":[...]}, {"input":[...]}, o lista directa.
    Devuelve SIEMPRE: [{"role":"...","content":"<string>"}]
    """
    if isinstance(data, dict) and "messages" in data:
        msgs = data["messages"]
    elif isinstance(data, dict) and "input" in data:
        msgs = data["input"]
    elif isinstance(data, list):
        msgs = data
    else:
        raise SystemExit("El JSON debe contener 'messages' o 'input' o ser una lista de mensajes.")

    out = []
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise SystemExit("Mensaje inválido: falta 'role' o 'content'.")
        role = m["role"]
        content = m["content"]
        if isinstance(content, list):
            text = "".join((p.get("text", "") if isinstance(p, dict) else str(p)) for p in content)
        else:
            text = str(content)
        out.append({"role": role, "content": text})
    return out

def extract_usage_tokens(r) -> Tuple[int, int]:
    usage = getattr(r, "usage", None)
    if usage is not None and hasattr(usage, "input_tokens"):
        return int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0)
    if isinstance(usage, dict):
        return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
    return 0, 0

def quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)

# -------------------- Cliente OpenAI --------------------
class OpenAIResponder:
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, input_payload: List[Dict[str, str]]) -> Tuple[str, int, int]:
        r = self.client.responses.create(model=self.model, input=input_payload)
        # texto
        text = getattr(r, "output_text", None) or ""
        if not text and getattr(r, "output", None):
            try:
                for item in r.output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        for c in item.get("content", []):
                            if isinstance(c, dict) and c.get("text"):
                                text = c["text"]
                                raise StopIteration
            except StopIteration:
                pass
        # tokens
        in_tok, out_tok = extract_usage_tokens(r)
        return text, in_tok, out_tok

# -------------------- Core (igual al “general”: loops por NPC) --------------------
async def run_async(
    prompt_json: str,
    model: str,
    minutes: int,
    npcs: int,
    interval_mean: float,
    price_in_per_1m: float,
    price_out_per_1m: float,
    plots: bool,
    show: bool,
):
    _use_backend(show)  # reconfigura si es necesario

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Falta OPENAI_API_KEY")

    with open(prompt_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    base_input = to_responses_input(raw)
    client = OpenAIResponder(api_key, model)

    start = time.time()
    deadline = start + 60 * minutes

    logs = []
    rng_global = random.Random(42)

    async def npc_loop(npc_id: int, seed: int):
        rng = random.Random(seed)
        while time.time() < deadline:
            # payload para este NPC
            input_payload = list(base_input) + [{
                "role": "user",
                "content": f"NPC {npc_id}: ejecuta tu próxima acción.",
            }]

            t0 = time.time()
            err = ""
            in_tok = out_tok = 0
            try:
                # llamar a OpenAI en thread pool para no bloquear el loop
                _, in_tok, out_tok = await asyncio.to_thread(client.complete, input_payload)
                ok = True
            except Exception as e:
                ok = False
                err = str(e)
            t1 = time.time()

            latency = (t1 - t0) if ok else float("nan")
            minute = int((t1 - start) // 60)

            logs.append({
                "npc_id": npc_id,
                "t_send": t0,
                "t_recv": t1,
                "latency_s": latency,
                "ok_json": 1 if ok else 0,
                "tokens_in": in_tok,
                "tokens_out": out_tok,
                "error": "" if ok else err,
                "minute": minute,
            })

            # igual que el general: intervalo exponencial por NPC con media interval_mean
            gap = rng.expovariate(1.0 / max(0.001, interval_mean))
            await asyncio.sleep(gap)

    tasks = [asyncio.create_task(npc_loop(i+1, rng_global.randint(1, 10_000_000))) for i in range(npcs)]
    await asyncio.gather(*tasks)

    # ---- DataFrame y CSVs
    df = pd.DataFrame(logs)
    df.to_csv("metrics_raw.csv", index=False, encoding="utf-8")

    df_ok = df[df["ok_json"] == 1].copy()
    latencies = df_ok["latency_s"].dropna().tolist()
    p50 = quantile(latencies, 0.50)
    p95 = quantile(latencies, 0.95)

    df_ok["tokens_total"] = df_ok["tokens_in"] + df_ok["tokens_out"]
    agg = df_ok.groupby("minute").agg(
        tokens_in_min=("tokens_in", "sum"),
        tokens_out_min=("tokens_out", "sum"),
        tokens_total_min=("tokens_total", "sum"),
        latency_mean_s=("latency_s", "mean"),
        latency_med_s=("latency_s", "median"),
        requests=("latency_s", "count"),
    ).reset_index()

    errs = df[df["ok_json"] == 0].groupby("minute").size().rename("errors").reset_index()
    agg = agg.merge(errs, on="minute", how="left")
    agg["errors"] = agg["errors"].fillna(0).astype(int)

    # costo por minuto (1M)
    if price_in_per_1m or price_out_per_1m:
        agg["cost_min_usd"] = (agg["tokens_in_min"]/1_000_000.0)*price_in_per_1m + (agg["tokens_out_min"]/1_000_000.0)*price_out_per_1m
    else:
        agg["cost_min_usd"] = 0.0

    agg.to_csv("agg_minute.csv", index=False, encoding="utf-8")

    # ---- Resumen (igual estilo)
    total_in = int(df_ok["tokens_in"].sum())
    total_out = int(df_ok["tokens_out"].sum())
    total_ok = int(df_ok.shape[0])
    total_err = int((df["ok_json"] == 0).sum())
    costo_total = (total_in/1_000_000.0)*price_in_per_1m + (total_out/1_000_000.0)*price_out_per_1m if (price_in_per_1m or price_out_per_1m) else None

    print("=== RESUMEN SESIÓN ===")
    print(f"Proveedor: openai | Modelo: {model}")
    print(f"NPCs: {npcs} | Duración: {minutes} min | Media intervalo: {interval_mean:.1f}s")
    print(f"Requests totales: {len(df)} | OK: {total_ok} | Errores: {total_err} | JSON inválido: 0")
    if latencies:
        print(f"Latencia p50: {p50:.3f}s | p95: {p95:.3f}s")
    else:
        print("Latencia p50: nans | p95: nans")
    print(f"Tokens totales (in+out): {int(df_ok['tokens_total'].sum())}")
    if costo_total is not None:
        print(f"Costo estimado (USD por 1M tokens): {costo_total:.6f} USD")
    else:
        print("Costo: N/A (sin precios configurados).")
    print("Guardado: metrics_raw.csv")
    print("Guardado: agg_minute.csv")

    # ---- Plots: SIEMPRE guardamos PNG; si --show, también abrimos ventana
    if not df_ok.empty:
        df_ok_sorted = df_ok.sort_values("t_recv").reset_index(drop=True)
        df_ok_sorted["idx"] = df_ok_sorted.index + 1

        # 1) Latencia por request
        plt.figure(figsize=(12, 4))
        plt.bar(df_ok_sorted["idx"], df_ok_sorted["latency_s"], width=0.8)
        if latencies:
            plt.axhline(p50, linestyle="--", label=f"p50={p50:.3f}s")
            plt.axhline(p95, linestyle="--", label=f"p95={p95:.3f}s")
            plt.legend()
        plt.title("Latencia por request")
        plt.xlabel("request # (orden temporal)")
        plt.ylabel("segundos")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_latency_per_request.png", dpi=150)
        if show: plt.show()
        plt.close()

        # 2) Tokens por request (apilados)
        plt.figure(figsize=(12, 4))
        plt.bar(df_ok_sorted["idx"], df_ok_sorted["tokens_in"], width=0.8, label="in")
        plt.bar(df_ok_sorted["idx"], df_ok_sorted["tokens_out"], width=0.8,
                bottom=df_ok_sorted["tokens_in"], label="out")
        plt.title("Tokens por request (in + out)")
        plt.xlabel("request # (orden temporal)")
        plt.ylabel("tokens")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_tokens_per_request.png", dpi=150)
        if show: plt.show()
        plt.close()

    # 3) Latencia por minuto (media y mediana)
    if not agg.empty:
        plt.figure(figsize=(10, 4))
        plt.plot(agg["minute"], agg["latency_med_s"], label="mediana/min")
        plt.plot(agg["minute"], agg["latency_mean_s"], label="media/min")
        if latencies:
            plt.axhline(p50, linestyle="--", label=f"p50 global={p50:.3f}s")
            plt.axhline(p95, linestyle="--", label=f"p95 global={p95:.3f}s")
        plt.title("Latencia por minuto")
        plt.xlabel("minuto")
        plt.ylabel("segundos")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_latency_per_min.png", dpi=150)
        if show: plt.show()
        plt.close()

        # 4) Tokens por minuto
        plt.figure(figsize=(10, 4))
        plt.plot(agg["minute"], agg["tokens_total_min"])
        plt.title("Tokens por minuto")
        plt.xlabel("minuto")
        plt.ylabel("tokens/min")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_tokens_per_min.png", dpi=150)
        if show: plt.show()
        plt.close()

        # 5) Costo por minuto
        plt.figure(figsize=(10, 4))
        plt.plot(agg["minute"], agg["cost_min_usd"])
        plt.title("Costo por minuto (USD)")
        plt.xlabel("minuto")
        plt.ylabel("USD/min")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_cost_per_min.png", dpi=150)
        if show: plt.show()
        plt.close()

def main():
    ap = argparse.ArgumentParser(description="Benchmark solo OpenAI (Responses API) — mismo ritmo que el general (loops por NPC).")
    ap.add_argument("--prompt_json", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--minutes", type=int, default=2)
    ap.add_argument("--npcs", type=int, default=5)
    ap.add_argument("--interval_mean", type=float, default=20.0, help="media del intervalo por NPC (segundos)")
    ap.add_argument("--price_in_per_1m", type=float, default=0.0)
    ap.add_argument("--price_out_per_1m", type=float, default=0.0)
    ap.add_argument("--plots", action="store_true", help="Generar y guardar gráficos PNG")
    ap.add_argument("--show", action="store_true", help="Mostrar ventanas además de guardar PNG")
    args = ap.parse_args()
    asyncio.run(run_async(
        prompt_json=args.prompt_json,
        model=args.model,
        minutes=args.minutes,
        npcs=args.npcs,
        interval_mean=args.interval_mean,
        price_in_per_1m=args.price_in_per_1m,
        price_out_per_1m=args.price_out_per_1m,
        plots=args.plots,
        show=args.show,
    ))

if __name__ == "__main__":
    main()
