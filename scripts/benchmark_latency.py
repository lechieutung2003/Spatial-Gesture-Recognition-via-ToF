import os
import sys
import time
import json
import numpy as np
import tensorflow as tf

def benchmark_tflite_model(model_path, num_warmup=50, num_runs=500):
    """Benchmarks a TFLite model measuring host latency and throughput."""
    if not os.path.exists(model_path):
        return None

    model_size_kb = os.path.getsize(model_path) / 1024.0
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Generate test sample
    if input_dtype == np.int8:
        dummy_input = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    # Benchmark loop
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # in ms

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    min_lat = float(np.min(latencies))
    max_lat = float(np.max(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "model_file": os.path.basename(model_path),
        "size_kb": round(model_size_kb, 2),
        "mean_ms": round(mean_lat, 3),
        "std_ms": round(std_lat, 3),
        "p50_ms": round(p50_lat, 3),
        "p95_ms": round(p95_lat, 3),
        "p99_ms": round(p99_lat, 3),
        "min_ms": round(min_lat, 3),
        "max_ms": round(max_lat, 3),
        "throughput_fps": round(fps, 1)
    }

def calculate_mcu_estimates(total_macs, params_count):
    """
    Computes realistic MCU execution metrics based on clock speed,
    cycles-per-MAC for DSP/SIMD extensions, and memory requirements.
    """
    # Hardware Profiles
    # ESP32-S3: 240 MHz Dual Core Xtensa LX7 with Vector Instructions (approx 3.5 cycles/MAC for float/int8 CNN+GRU)
    # STM32H743: 480 MHz Cortex-M7 with double-precision FPU & DSP (approx 2.2 cycles/MAC)
    # STM32F401: 84 MHz Cortex-M4 with FPU (approx 5.5 cycles/MAC)
    
    profiles = {
        "ESP32-S3 (240MHz, Vector DSP)": {
            "clock_mhz": 240,
            "cycles_per_mac": 3.8,
            "preprocess_ms": 0.22,
            "tensor_arena_kb": 38.0
        },
        "STM32H743 (480MHz, Cortex-M7)": {
            "clock_mhz": 480,
            "cycles_per_mac": 2.4,
            "preprocess_ms": 0.11,
            "tensor_arena_kb": 38.0
        },
        "STM32F446 (180MHz, Cortex-M4)": {
            "clock_mhz": 180,
            "cycles_per_mac": 5.2,
            "preprocess_ms": 0.35,
            "tensor_arena_kb": 38.0
        }
    }

    sensor_fps = 15.0
    frame_budget_ms = 1000.0 / sensor_fps # 66.67 ms

    results = {}
    for name, spec in profiles.items():
        inference_cycles = total_macs * spec["cycles_per_mac"]
        inference_ms = (inference_cycles / (spec["clock_mhz"] * 1e6)) * 1000.0
        total_pipeline_ms = inference_ms + spec["preprocess_ms"]
        headroom_pct = ((frame_budget_ms - total_pipeline_ms) / frame_budget_ms) * 100.0
        max_fps = 1000.0 / total_pipeline_ms

        results[name] = {
            "clock_mhz": spec["clock_mhz"],
            "inference_latency_ms": round(inference_ms, 2),
            "preprocess_latency_ms": round(spec["preprocess_ms"], 2),
            "total_pipeline_ms": round(total_pipeline_ms, 2),
            "sensor_frame_budget_ms": round(frame_budget_ms, 2),
            "headroom_pct": round(headroom_pct, 1),
            "max_supported_fps": round(max_fps, 1),
            "realtime_capable": total_pipeline_ms < frame_budget_ms,
            "tensor_arena_ram_kb": spec["tensor_arena_kb"]
        }
    return results

def main():
    print("=" * 80)
    print("SPATIAL GESTURE RECOGNITION - INFERENCE BENCHMARK & PROFILER")
    print("=" * 80)

    # 1. Host Latency Benchmarks
    models = [
        "models/gesture_model_f32.tflite",
        "models/gesture_model_f16.tflite",
        "models/gesture_model_int8.tflite"
    ]

    host_results = []
    for m in models:
        if os.path.exists(m):
            print(f"Benchmarking {m} (50 warmup, 500 timed runs)...")
            res = benchmark_tflite_model(m, num_warmup=50, num_runs=500)
            if res:
                host_results.append(res)

    print("\n--- HOST CPU BENCHMARK RESULTS ---")
    print(f"{'Model':<28} | {'Flash (KB)':<10} | {'Mean (ms)':<10} | {'P50 (ms)':<9} | {'P95 (ms)':<9} | {'Throughput':<12}")
    print("-" * 88)
    for r in host_results:
        print(f"{r['model_file']:<28} | {r['size_kb']:<10.1f} | {r['mean_ms']:<10.3f} | {r['p50_ms']:<9.3f} | {r['p95_ms']:<9.3f} | {r['throughput_fps']:<8.1f} FPS")

    # 2. Total MACs calculation for CNN-GRU
    # CNN 1: 20 * 8 * 8 * (3*3*1*8) = 92,160 MACs
    # MaxPool 1: subsampling
    # CNN 2: 20 * 4 * 4 * (3*3*8*16) = 368,640 MACs
    # MaxPool 2: subsampling
    # GRU: 20 timesteps * 3 * (64*32 + 32*32) = 184,320 MACs
    # Dense 1: 32 * 16 = 512 MACs
    # Dense 2: 16 * 7 = 112 MACs
    total_macs = 92160 + 368640 + 184320 + 512 + 112 # ~645,744 MACs
    total_params = 11303

    mcu_results = calculate_mcu_estimates(total_macs, total_params)

    print("\n--- EMBEDDED TARGET PROFILING & REAL-TIME BUDGET (15 Hz / 66.7 ms Frame Budget) ---")
    print(f"{'Target MCU':<32} | {'Preproc':<8} | {'Inference':<10} | {'Pipeline':<9} | {'Headroom':<10} | {'Status':<12}")
    print("-" * 92)
    for target, spec in mcu_results.items():
        status = "PASS (Real-Time)" if spec["realtime_capable"] else "FAIL"
        print(f"{target:<32} | {spec['preprocess_latency_ms']:<6.2f}ms | {spec['inference_latency_ms']:<8.2f}ms | {spec['total_pipeline_ms']:<7.2f}ms | {spec['headroom_pct']:<8.1f}% | {status}")

    # Save summary
    benchmark_data = {
        "host_benchmark": host_results,
        "model_complexity": {
            "total_parameters": total_params,
            "total_macs": total_macs,
            "model_memory_flash_kb": {r["model_file"]: r["size_kb"] for r in host_results}
        },
        "mcu_embedded_profiling": mcu_results
    }

    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmarking_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print("\nSaved benchmark results to docs/benchmarking_results.json\n")

if __name__ == "__main__":
    main()
