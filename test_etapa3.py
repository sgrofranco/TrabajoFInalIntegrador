import json
import math
import pandas as pd
import re
import matplotlib.pyplot as plt
from typing import List

def p_quantile(values: List[float], q: float) -> float:
    """Calcula el percentil exacto de una lista de valores."""
    if not values: return float("nan")
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] if lo == hi else s[lo] + (s[hi]-s[lo])*(pos-lo)

def extraer_y_validar_json(texto: str) -> bool:
    """Limpia el markdown y valida que tenga las claves obligatorias."""
    if not texto: return False
    texto_limpio = re.sub(r'^```json\s*|\s*```$', '', texto.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(texto_limpio)
        if isinstance(data, dict) and "Accion" in data and "Dialogo" in data:
            return True
        return False
    except Exception:
        return False

def main():
    archivo_entrada = "logs (6).json"
    
    try:
        with open(archivo_entrada, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        print(f"No se encontró el archivo {archivo_entrada}")
        return

    if not datos:
        print("El archivo JSON está vacío.")
        return

    # Extraer el nombre del modelo del primer registro para los títulos de los gráficos
    modelo_nombre = datos[0].get("model", "modelo-desconocido")

    # Usamos el timestamp de la primera petición como minuto 0
    t0 = min([item.get("created", 0) for item in datos])
    
    registros = []
    
    for idx, item in enumerate(datos):
        created = item.get("created", 0)
        minuto = int((created - t0) // 60)
        
        # Uso de tokens
        uso = item.get("usage", {})
        tokens_in = uso.get("prompt_tokens", 0)
        tokens_out = uso.get("completion_tokens", 0)
        tokens_totales = uso.get("total_tokens", tokens_in + tokens_out)
        
        # Latencia (pasando de milisegundos a segundos)
        latencia_ms = item.get("Latency", 0)
        latency_s = latencia_ms / 1000.0
        
        # Validar JSON
        contenido = ""
        choices = item.get("choices", [])
        if choices:
            contenido = choices[0].get("message", {}).get("content", "")
            
        json_valido = 1 if extraer_y_validar_json(contenido) else 0
        
        registros.append({
            "request_idx": idx, # Empezamos en 0 para coincidir con tus gráficos
            "timestamp": created,
            "minute": minuto,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_totales,
            "ok_json": json_valido,
            "latency_s": latency_s
        })

    # --- 1. Exportar Raw Metrics ---
    df = pd.DataFrame(registros)

    # --- PARCHE PARA LIMPIAR LATENCIAS ANÓMALAS ---
    # 1. Calculamos la mediana solo con los valores normales (menores a 100 segundos)
    mediana_latencia = df.loc[df["latency_s"] < 100, "latency_s"].median()

    # 2. Reemplazamos cualquier locura mayor a 100 segundos por esa mediana
    df["latency_s"] = df["latency_s"].apply(lambda x: x if x < 100 else mediana_latencia)
    # ----------------------------------------------

    df.to_csv("metrics_raw_etapa3.csv", index=False, encoding="utf-8")
    
    # --- 2. Cálculos globales y Percentiles ---
    df_ok = df[df["ok_json"] == 1]
    latencias = df_ok["latency_s"].tolist()
    p50 = p_quantile(latencias, 0.50)
    p95 = p_quantile(latencias, 0.95)
    
    # --- 3. Agregación por minuto ---
    agg = df.groupby("minute").agg(
        tokens_in_min=("tokens_in", "sum"),
        tokens_out_min=("tokens_out", "sum"),
        tokens_total_min=("tokens_total", "sum"),
        latency_mean_s=("latency_s", "mean"),
        latency_med_s=("latency_s", "median"),
        requests=("timestamp", "count"),
        ok_json=("ok_json", "sum")
    ).reset_index()
    
    agg["invalid_json"] = agg["requests"] - agg["ok_json"]
    
    # Costo por minuto (puedes ajustar los precios si es necesario)
    precio_in_por_1m = 0.0
    precio_out_por_1m = 0.0
    agg["cost_min_usd"] = (agg["tokens_in_min"]/1_000_000.0)*precio_in_por_1m + (agg["tokens_out_min"]/1_000_000.0)*precio_out_por_1m
    
    agg.to_csv("agg_minute_etapa3.csv", index=False, encoding="utf-8")

    # --- 4. Generación de Gráficos (Estilo Etapa 1) ---
    
    # Gráfico 1: Latencia por minuto
    plt.figure(figsize=(10, 4))
    plt.plot(agg["minute"], agg["latency_med_s"], label="mediana/min", color="C0")
    plt.plot(agg["minute"], agg["latency_mean_s"], label="media/min", color="C1")
    if latencias:
        plt.axhline(p50, linestyle="--", color="C0", label=f"p50 global = {p50:.3f}s")
        plt.axhline(p95, linestyle="--", color="C0", label=f"p95 global = {p95:.3f}s")
    plt.title(f"Latencia por minuto — {modelo_nombre}")
    plt.xlabel("minuto")
    plt.ylabel("segundos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_latency_per_min_etapa3.png", dpi=150)
    plt.close()

    # Gráfico 2: Tokens por minuto
    plt.figure(figsize=(10, 4))
    plt.plot(agg["minute"], agg["tokens_total_min"], color="C0")
    plt.title(f"Tokens/min (sum NPCs) — {modelo_nombre}")
    plt.xlabel("minuto")
    plt.ylabel("tokens/min")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_tokens_per_min_etapa3.png", dpi=150)
    plt.close()

    # Gráfico 3: Latencia por request
    plt.figure(figsize=(12, 4))
    plt.bar(df["request_idx"], df["latency_s"], width=0.8, color="C0")
    if latencias:
        plt.axhline(p50, linestyle="--", color="C1", label=f"p50 global = {p50:.3f}s") # Naranja
        plt.axhline(p95, linestyle="--", color="C3", label=f"p95 global = {p95:.3f}s") # Rojo
    plt.title(f"Latencia por request — {modelo_nombre}")
    plt.xlabel("request # (orden temporal)")
    plt.ylabel("segundos")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_latency_per_request_etapa3.png", dpi=150)
    plt.close()

    # Gráfico 4: Tokens por request (in + out)
    plt.figure(figsize=(12, 4))
    plt.bar(df["request_idx"], df["tokens_in"], width=0.8, color="C0")
    plt.bar(df["request_idx"], df["tokens_out"], width=0.8, bottom=df["tokens_in"], color="C1")
    plt.title(f"Tokens por request (in + out) — {modelo_nombre}")
    plt.xlabel("request # (orden temporal)")
    plt.ylabel("tokens")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_tokens_per_request_etapa3.png", dpi=150)
    plt.close()

    # Gráfico 5 (Opcional): Costo por minuto
    if precio_in_por_1m > 0 or precio_out_por_1m > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(agg["minute"], agg["cost_min_usd"], color="C0")
        plt.title(f"Costo/min — {modelo_nombre}")
        plt.xlabel("minuto")
        plt.ylabel("USD/min")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_cost_per_min_etapa3.png", dpi=150)
        plt.close()

    print("\n=== RESUMEN SESIÓN - ETAPA 3 ===")
    print(f"Modelo detectado: {modelo_nombre}")
    print(f"Requests totales: {len(df)} | OK: {df_ok.shape[0]}")
    print(f"Latencia p50: {p50:.3f}s | p95: {p95:.3f}s")
    print(f"Tokens totales (in+out): {df['tokens_total'].sum()}")
    print("Exportación completa: CSVs y los 4 gráficos estilo 'Etapa 1' han sido generados.")

if __name__ == "__main__":
    main()