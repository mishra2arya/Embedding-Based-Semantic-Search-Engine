# Performance & Load Testing Benchmarks

## Benchmark Methodology
High-concurrency stress testing measures system behavior under load levels:
`10, 50, 100, 250, 500` concurrent worker threads.

### Measured Metrics
- **P50 Latency (ms)**: Median query duration.
- **P90 / P95 Latency (ms)**: Tail latencies representing SLA targets.
- **P99 Latency (ms)**: Extreme tail latency under lock or cache contention.
- **Throughput (Requests/sec)**: Max sustainable request capacity.
- **Error Rate**: Ratio of HTTP 5xx / connection timeouts.

### Execution
```bash
python scripts/benchmark.py --concurrency 10,50,100,250,500 --output benchmarks
# or using Make
make benchmark
```
Outputs are written to:
- `benchmarks/benchmark.csv`
- `benchmarks/benchmark.json`
- `benchmarks/benchmark.md`
