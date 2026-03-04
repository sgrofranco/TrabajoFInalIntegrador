import os, json, time, math, argparse, asyncio, random
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import pandas as pd
from matplotlib import pyplot as plt
import google.generativeai as genai

# ========= utilidades métricas =========
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

# ========= carga y normalización del prompt JSON =========
def load_messages_any(prompt_path: str) -> List[Dict[str, Any]]:
    """
    Acepta:
      A) OpenAI-style: {"messages":[{"role","content"}, ...]}
      B) Gemini nativo: {"system_instruction": {...}, "contents":[...], "generationConfig": {...}}
    Devuelve siempre messages OpenAI-style.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if "messages" in payload and isinstance(payload["messages"], list):
        return payload["messages"]

    if "contents" in payload:  # Gemini nativo -> convertir
        messages: List[Dict[str, Any]] = []

        # 1) system_instruction -> role=system
        sys = payload.get("system_instruction", {})
        sys_parts = sys.get("parts", [])
        sys_texts = []
        for p in sys_parts:
            if isinstance(p, dict) and "text" in p and str(p["text"]).strip():
                sys_texts.append(str(p["text"]).strip())
        sys_text = "\n\n".join(sys_texts).strip()
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

        # 2) contents -> role user/assistant
        for item in payload.get("contents", []):
            role = item.get("role", "user")
            parts = item.get("parts", [])
            texts = []
            for p in parts:
                if isinstance(p, dict) and "text" in p and str(p["text"]).strip():
                    texts.append(str(p["text"]).strip())
            text = "\n\n".join(texts).strip()
            if not text:
                continue
            mapped_role = "user" if role == "user" else "assistant"
            messages.append({"role": mapped_role, "content": text})

        return messages

    raise SystemExit("El JSON debe tener 'messages' (OpenAI-style) o 'contents' (Gemini nativo).")

def split_system_and_history(messages: List[Dict[str, Any]]):
    """Gemini: system_instruction (str|None) + history (role user/model) - history NO se usa con generate_content,
    pero mantenemos por compatibilidad si quisieras cambiar a chat."""
    systems, history = [], []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        if role == "system":
            systems.append(str(content))
        elif role in ("user", "assistant"):
            gr = "user" if role == "user" else "model"
            history.append({"role": gr, "parts": [str(content)]})
        else:
            history.append({"role": "user", "parts": [str(content)]})
    sys_instr = "\n\n".join([s for s in systems if s.strip()]) if systems else None
    return sys_instr, history

# ========= Adapter Gemini =========
class GeminiAdapter:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model_name = model_name

    def _make_model(self, system_instruction: Optional[str]):
        return genai.GenerativeModel(
            self.model_name,
            system_instruction=system_instruction if system_instruction else None,
            generation_config={"response_mime_type": "application/json"},
        )

    def _extract_prompt_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        # 1) último user no vacío
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, (dict, list)):
                    c = json.dumps(c, ensure_ascii=False, indent=2)
                if c and str(c).strip():
                    return str(c).strip()
        # 2) fallback: concatenación de todos los contenidos no vacíos
        parts = []
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, (dict, list)):
                c = json.dumps(c, ensure_ascii=False, indent=2)
            if c and str(c).strip():
                parts.append(str(c).strip())
        joined = "\n\n".join(parts).strip()
        return joined

    def _usage_counts(self, response):
        try:
            um = getattr(response, "usage_metadata", None)
            if um:
                pt = getattr(um, "prompt_token_count", None)
                ct = getattr(um, "candidates_token_count", None)
                tt = getattr(um, "total_token_count", None)
                return (int(pt) if pt is not None else None,
                        int(ct) if ct is not None else None,
                        int(tt) if tt is not None else None)
        except Exception:
            pass
        return None, None, None

    def _response_text(self, response) -> str:
        try:
            return response.text or ""
        except Exception:
            return ""

    def _complete_sync(self, messages: List[Dict[str, Any]]):
        system_instruction, _history = split_system_and_history(messages)
        model = self._make_model(system_instruction)

        prompt_text = self._extract_prompt_from_messages(messages).strip()
        if not prompt_text:
            # Nunca mandar vacío; salvavidas:
            prompt_text = 'Devuelve SOLO un JSON válido con la forma {"accion":"...","dialogo":"..."}'
        # Para evitar el error de contenido vacío: usamos generate_content(prompt_text)
        return model.generate_content(prompt_text)

    async def complete(self, messages: List[Dict[str, Any]]):
        return await asyncio.to_thread(self._complete_sync, messages)

# ========= CLI =========
def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--prompt_json", required=True)
    p.add_argument("--npcs", type=int, default=5)
    p.add_argument("--minutes", type=float, default=1.0)
    p.add_argument("--mean_interval", type=float, default=20.0)
    p.add_argument("--bucket_sec", type=int, default=60)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--price_in_per_1k", type=float, default=float(os.environ.get("PRICE_IN_PER_1K", "0")))
    p.add_argument("--price_out_per_1k", type=float, default=float(os.environ.get("PRICE_OUT_PER_1K", "0")))
    return p.parse_args()

def get_google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        raise SystemExit("Falta GOOGLE_API_KEY")
    return key

# ========= main =========
async def main():
    args = build_args()
    MESSAGES = load_messages_any(args.prompt_json)  # acepta ambos formatos

    adapter = GeminiAdapter(get_google_api_key(), args.model)

    results: List[Record] = []
    t0 = now_s()
    end_time = t0 + args.minutes * 60

    async def npc_loop(npc_id: int, seed: int):
        rng = random.Random(seed)
        def sample_interval():
            lam = 1.0 / max(0.01, args.mean_interval)
            return rng.expovariate(lam)

        while now_s() < end_time:
            payload_messages = MESSAGES  # son inmutables; si vas a mutar, haz deepcopy
            est_tokens_in = approx_token_count(json.dumps(payload_messages, ensure_ascii=False))

            t_send = now_s()
            tokens_in, tokens_out = est_tokens_in, 0
            latency, ok_flag = None, 0

            try:
                resp = await adapter.complete(payload_messages)
                t_recv = now_s()
                latency = t_recv - t_send

                content = adapter._response_text(resp)
                data = extract_json_object(content)
                ok_flag = 1 if data is not None else 0

                pt, ct, tt = adapter._usage_counts(resp)
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
                    latency_s=None, ok_json=0,
                    tokens_in=est_tokens_in, tokens_out=0,
                    error=str(e), minute=minute_bucket(t0, t_recv, args.bucket_sec)
                ))
                await asyncio.sleep(min(10.0, 1.0 + random.random()*2.0))
            else:
                await asyncio.sleep(sample_interval())

    tasks = [asyncio.create_task(npc_loop(i, args.seed + i + 1)) for i in range(args.npcs)]
    await asyncio.gather(*tasks)

    # ----- DataFrame + agregados -----
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
        requests=("latency_s", "count"),
    ).reset_index()

    errs = df[df["error"].notna()].groupby("minute").size().rename("errors").reset_index()
    invalid = df_ok[df_ok["ok_json"] == 0].groupby("minute").size().rename("invalid_json").reset_index()
    agg = agg.merge(errs, on="minute", how="left").merge(invalid, on="minute", how="left")
    agg["errors"] = agg["errors"].fillna(0).astype(int)
    agg["invalid_json"] = agg["invalid_json"].fillna(0).astype(int)

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
    print(f"Proveedor: google | Modelo: {args.model}")
    print(f"NPCs: {args.npcs} | Duración: {args.minutes} min | Media intervalo: {args.mean_interval:.1f}s")
    print(f"Requests totales: {total_reqs} | OK: {total_ok} | Errores: {total_err} | JSON inválido: {total_invalid}")
    print(f"Latencia p50: {p50:.3f}s | p95: {p95:.3f}s")
    print(f"Tokens totales (in+out): {int(df_ok['tokens_total'].sum())}")
    if PRICE_IN > 0 or PRICE_OUT > 0:
        print(f"Costo total estimado: ${agg['cost_min_usd'].sum():.4f} USD")
    else:
        print("Costo: N/A (sin precios configurados).")

    if args.plots and not agg.empty:
        # Tokens/min
        plt.figure()
        plt.plot(agg["minute"], agg["tokens_total_min"])
        plt.title(f"Tokens/min (sum {args.npcs} NPCs) — {args.model}")
        plt.xlabel("minuto"); plt.ylabel("tokens/min"); plt.grid(True, alpha=0.3)
        plt.savefig("plot_tokens_per_min.png", dpi=150, bbox_inches="tight")

        # Latencia/min + p50/p95 global
        plt.figure()
        plt.plot(agg["minute"], agg["latency_med_s"], label="mediana/min")
        plt.plot(agg["minute"], agg["latency_mean_s"], label="media/min")
        plt.axhline(p50, linestyle="--", label=f"p50 global = {p50:.3f}s")
        plt.axhline(p95, linestyle="--", label=f"p95 global = {p95:.3f}s")
        plt.title(f"Latencia por minuto — {args.model}")
        plt.xlabel("minuto"); plt.ylabel("segundos"); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig("plot_latency_per_min.png", dpi=150, bbox_inches="tight")

        # Costo/min
        plt.figure()
        plt.plot(agg["minute"], agg["cost_min_usd"])
        plt.title(f"Costo/min — {args.model}")
        plt.xlabel("minuto"); plt.ylabel("USD/min"); plt.grid(True, alpha=0.3)
        plt.savefig("plot_cost_per_min.png", dpi=150, bbox_inches="tight")

        # Por request (barras finas)
        df_ok_sorted = df_ok.sort_values("t_recv").reset_index(drop=True)
        df_ok_sorted["request_idx"] = df_ok_sorted.index

        # Latencia por request
        plt.figure(figsize=(12, 4))
        plt.bar(df_ok_sorted["request_idx"], df_ok_sorted["latency_s"], width=0.8)
        plt.title(f"Latencia por request — {args.model}")
        plt.xlabel("request # (orden temporal)"); plt.ylabel("segundos")
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
        plt.title(f"Tokens por request (in + out) — {args.model}")
        plt.xlabel("request # (orden temporal)"); plt.ylabel("tokens")
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig("plot_tokens_per_request.png", dpi=150, bbox_inches="tight")

if __name__ == "__main__":
    asyncio.run(main())
