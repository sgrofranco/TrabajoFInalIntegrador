# benchmark_multi_provider.py
# Simulación de N NPCs contra Mistral / Groq / Cerebras con intervalos independientes,
# registro de latencia, validación JSON, tokens, agregación por minuto y p50/p95 + costo/min opcional.
#
# Uso:
#   export MISTRAL_API_KEY=...   # si provider = mistral
#   export GROQ_API_KEY=...      # si provider = groq
#   export CEREBRAS_API_KEY=...  # si provider = cerebras (API OpenAI-compatible)
#   python test_benchmark_npcs.py --provider groq --prompt_json prompt_json.json --model llama-3.1-70b-versatile --minutes 5 --plots
#
# Nota: basado en tu script original (mismas métricas/plots), pero con adaptadores por proveedor.
#       Franco Sgro — 2025-09-29

import os, json, time, math, argparse, asyncio, random
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
import pandas as pd
from matplotlib import pyplot as plt

# --- Utilidades métricas (idénticas a tu versión) ---
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
    txt = text.strip()
    try:
        data = json.loads(txt)
        if isinstance(data, dict) and "accion" in data and "dialogo" in data:
            return data
    except Exception:
        pass
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

# --- Adapter de proveedores ---
class ProviderAdapter:
    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.api_key = api_key
        self.model = model

        if provider == "mistral":
            from mistralai import Mistral
            self.client = Mistral(api_key=api_key)
            self.kind = "mistral"
        elif provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.kind = "groq"
        elif provider == "cerebras":
            # API OpenAI-compatible (base_url de Cerebras)
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url="https://api.cerebras.ai/v1")
            self.kind = "openai_like"
        else:
            raise SystemExit(f"Proveedor no soportado: {provider}")

    def _normalize_messages(self, payload_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Asegurar que content sea string (coincide con tu normalización)
        out = []
        for m in payload_messages:
            m = deepcopy(m)
            if isinstance(m.get("content"), (dict, list)):
                m["content"] = json.dumps(m["content"], ensure_ascii=False, indent=2)
            out.append(m)
        return out

    def _parse_usage(self, resp) -> (Optional[int], Optional[int], Optional[int]):
        """
        Devuelve (prompt_tokens, completion_tokens, total_tokens) si están presentes.
        Soporta: mistral, groq y openai-like.
        """
        try:
            usage = getattr(resp, "usage", None) or (hasattr(resp, "model_dump") and resp.model_dump().get("usage"))
            if usage is None and isinstance(resp, dict):
                usage = resp.get("usage")

            if usage is None:
                return None, None, None

            # usage puede ser objeto con attrs o dict:
            def getu(obj, key):
                return getattr(obj, key, None) if not isinstance(obj, dict) else obj.get(key)

            pt = getu(usage, "prompt_tokens")
            ct = getu(usage, "completion_tokens")
            tt = getu(usage, "total_tokens")
            return pt if isinstance(pt, int) else None, \
                   ct if isinstance(ct, int) else None, \
                   tt if isinstance(tt, int) else None
        except Exception:
            return None, None, None

    def _first_message_content(self, resp) -> str:
        # Unifica obtención del contenido
        try:
            # mistral SDK: resp.choices[0].message.content
            ch = getattr(resp, "choices", None)
            if ch:
                msg = ch[0].message if hasattr(ch[0], "message") else ch[0]
                content = getattr(msg, "content", None) or (isinstance(msg, dict) and msg.get("content"))
                if isinstance(content, list):
                    # algunos openai-like pueden devolver list de partes
                    content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
                return content or ""
        except Exception:
            pass

        # OpenAI-like dict fallback
        if isinstance(resp, dict):
            ch = resp.get("choices")
            if ch:
                msg = ch[0].get("message", {})
                return msg.get("content", "") or ""
        return ""

    def _make_request_sync(self, messages: List[Dict[str, Any]]):
        msgs = self._normalize_messages(messages)
        if self.kind == "mistral":
            # mistralai: client.chat.complete(model=..., messages=[{role, content}, ...])
            return self.client.chat.complete(model=self.model, messages=msgs)
        elif self.kind == "groq":
            # groq (OpenAI-like): client.chat.completions.create(model=..., messages=[...])
            return self.client.chat.completions.create(model=self.model, messages=msgs)
        else:
            # openai-like (Cerebras)
            return self.client.chat.completions.create(model=self.model, messages=msgs)

    async def complete(self, messages: List[Dict[str, Any]]):
        # Llamamos en thread para no bloquear el event loop (como en tu versión)
        return await asyncio.to_thread(self._make_request_sync, messages)

# --- CLI ---
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["mistral", "groq", "cerebras"], required=True)
    p.add_argument("--prompt_json", required=True, help="Ruta al JSON con 'messages' y opcionalmente 'model'.")
    p.add_argument("--model", default=None, help="Modelo a usar (prioriza sobre el JSON).")
    p.add_argument("--npcs", type=int, default=5)
    p.add_argument("--minutes", type=int, default=1)
    p.add_argument("--mean_interval", type=float, default=20.0, help="Media de intervalo (seg) por NPC.")
    p.add_argument("--bucket_sec", type=int, default=60, help="Bucket de agregación (seg).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plots", action="store_true")
    # Precios opcionales para costo estimado
    p.add_argument("--price_in_per_1k", type=float, default=float(os.environ.get("PRICE_IN_PER_1K", "0")))
    p.add_argument("--price_out_per_1k", type=float, default=float(os.environ.get("PRICE_OUT_PER_1K", "0")))
    return p.parse_args()

# --- Defaults por proveedor (puede variar; override con --model) ---
DEFAULT_MODELS = {
    "mistral":  "mistral-large-latest",
    "groq":     "llama-3.1-70b-versatile",
    "cerebras": "llama3.1-70b",
}

def get_api_key(provider: str) -> str:
    env = {
        "mistral":  "MISTRAL_API_KEY",
        "groq":     "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
    }[provider]
    key = os.environ.get(env, "").strip()
    if not key:
        raise SystemExit(f"Falta {env}")
    return key

async def main():
    args = build_args()

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        base_payload = json.load(f)
    MESSAGES = base_payload.get("messages")
    if not isinstance(MESSAGES, list):
        raise SystemExit("El JSON debe contener 'messages' (lista).")

    MODEL = args.model or base_payload.get("model") or DEFAULT_MODELS[args.provider]
    API_KEY = get_api_key(args.provider)

    adapter = ProviderAdapter(args.provider, API_KEY, MODEL)

    results: List[Record] = []
    t0 = now_s()
    end_time = t0 + args.minutes * 60

    async def npc_loop(npc_id: int, seed: int):
        rng = random.Random(seed)
        def sample_interval():
            lam = 1.0 / max(0.01, args.mean_interval)
            return rng.expovariate(lam)

        while now_s() < end_time:
            payload_messages = deepcopy(MESSAGES)
            est_tokens_in = approx_token_count(json.dumps(payload_messages, ensure_ascii=False))

            t_send = now_s()
            tokens_in, tokens_out = est_tokens_in, 0
            latency, ok_flag = None, 0

            try:
                resp = await adapter.complete(payload_messages)
                t_recv = now_s()
                latency = t_recv - t_send

                content = adapter._first_message_content(resp)
                data = extract_json_object(content)
                ok_flag = 1 if data is not None else 0

                # usage (prompt/completion/total)
                pt, ct, tt = adapter._parse_usage(resp)
                if isinstance(pt, int): tokens_in = pt
                if isinstance(ct, int): tokens_out = ct
                if not isinstance(pt, int) and isinstance(tt, int):
                    tokens_in = max(0, int(tt * 0.7))
                    tokens_out = tt - tokens_in
                if not isinstance(ct, int) and not isinstance(tt, int):
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

    # --- DataFrame y agregados (igual a tu flujo) ---
    df = pd.DataFrame([asdict(r) for r in results])
    if df.empty:
        print("No se registraron resultados."); return
    df.to_csv("metrics_raw.csv", index=False, encoding="utf-8")

    df_ok = df[df["latency_s"].notna()].copy()
    latencies = df_ok["latency_s"].tolist()
    p50, p95 = p_quantile(latencies, 0.50), p_quantile(latencies, 0.95)

    df_ok["tokens_total"] = df_ok["tokens_in"] + df_ok["tokens_out"]
    agg = df_ok.groupby("minute").agg(
        tokens_in_min=("tokens_in", "sum"),
        tokens_out_min=("tokens_out", "sum"),
        tokens_total_min=("tokens_total", "sum"),
        latency_mean_s=("latency_s", "mean"),
        latency_med_s=("latency_s", "median"),
        requests=("latency_s", "count")
    ).reset_index()

    errs = df[df["error"].notna()].groupby("minute").size().rename("errors").reset_index()
    invalid = df_ok[df_ok["ok_json"] == 0].groupby("minute").size().rename("invalid_json").reset_index()
    agg = agg.merge(errs, on="minute", how="left").merge(invalid, on="minute", how="left")
    agg["errors"] = agg["errors"].fillna(0).astype(int)
    agg["invalid_json"] = agg["invalid_json"].fillna(0).astype(int)

    # Costos (si configurás precios)
    PRICE_IN = args.price_in_per_1k
    PRICE_OUT = args.price_out_per_1k
    if PRICE_IN > 0 or PRICE_OUT > 0:
        agg["cost_min_usd"] = (agg["tokens_in_min"]/1000.0)*PRICE_IN + (agg["tokens_out_min"]/1000.0)*PRICE_OUT
    else:
        agg["cost_min_usd"] = 0.0

    agg.to_csv("agg_minute.csv", index=False, encoding="utf-8")

    total_reqs = len(df)
    total_ok = df_ok.shape[0]
    total_err = int(df["error"].notna().sum())
    total_invalid = int((df_ok["ok_json"] == 0).sum())

    print("\n=== RESUMEN SESIÓN ===")
    print(f"Proveedor: {args.provider} | Modelo: {MODEL}")
    print(f"NPCs: {args.npcs} | Duración: {args.minutes} min | Media intervalo: {args.mean_interval:.1f}s")
    print(f"Requests totales: {total_reqs} | OK: {total_ok} | Errores: {total_err} | JSON inválido: {total_invalid}")
    print(f"Latencia p50: {p50:.3f}s | p95: {p95:.3f}s")
    print(f"Tokens totales (in+out): {int(df_ok['tokens_total'].sum())}")
    if PRICE_IN > 0 or PRICE_OUT > 0:
        print(f"Costo total estimado: ${agg['cost_min_usd'].sum():.4f} USD")
    else:
        print("Costo: N/A (sin precios configurados).")

    if args.plots:
        # Tokens/min
        plt.figure()
        plt.plot(agg["minute"], agg["tokens_total_min"])
        plt.title(f"Tokens/min (sum {args.npcs} NPCs) — {MODEL}")
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

        # --- NUEVOS: por request (barras finas), como en tu última iteración ---
        df_ok_sorted = df_ok.sort_values("t_recv").reset_index(drop=True).copy()
        df_ok_sorted["request_idx"] = df_ok_sorted.index

        # Latencia por request
        plt.figure(figsize=(12, 4))
        plt.bar(df_ok_sorted["request_idx"], df_ok_sorted["latency_s"], width=0.8)
        plt.title(f"Latencia por request — {MODEL}")
        plt.xlabel("request # (orden temporal)")
        plt.ylabel("segundos")
        plt.grid(True, axis="y", alpha=0.3)
        plt.axhline(p50, linestyle="--", label=f"p50 global = {p50:.3f}s")
        plt.axhline(p95, linestyle="--", label=f"p95 global = {p95:.3f}s")
        plt.legend()
        plt.savefig("plot_latency_per_request.png", dpi=150, bbox_inches="tight")

        # Tokens por request (apilados in/out)
        plt.figure(figsize=(10, 4))
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
