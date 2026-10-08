/**
 * VectorIQ — Zero-Dependency SVG Charting Library
 * Production-grade, high-performance data visualizations with 100% offline support.
 */

const Charts = {
  /**
   * Render a horizontal latency waterfall breakdown chart.
   * @param {HTMLElement} container
   * @param {Object} breakdown - { embedding_ms, faiss_search_ms, metadata_fetch_ms, lexical_search_ms, rerank_ms, total_ms }
   */
  renderLatencyWaterfall(container, breakdown) {
    if (!container) return;
    if (!breakdown || Object.keys(breakdown).length === 0) {
      container.innerHTML = `<div class="chart-empty">No latency profile recorded for this query.</div>`;
      return;
    }

    const steps = [
      { key: "embedding_generation_ms", label: "Query Embedding", color: "#c7a5ff" },
      { key: "faiss_retrieval_ms", label: "FAISS HNSW Search", color: "#79a2f6" },
      { key: "lexical_search_ms", label: "BM25 Lexical Search", color: "#009bdc" },
      { key: "metadata_fetch_ms", label: "SQLite Metadata & Filter", color: "#008fb5" },
      { key: "reranking_ms", label: "Alignment Reranking", color: "#007f88" },
      { key: "context_assembly_ms", label: "Context & Serialization", color: "#006c5c" },
    ];

    const activeSteps = steps
      .map((s) => ({
        ...s,
        val: Math.max(0.01, breakdown[s.key] || breakdown[s.key.replace("_generation_ms", "_ms")] || 0),
      }))
      .filter((s) => s.val > 0.001);

    const total = breakdown.total_ms || activeSteps.reduce((acc, s) => acc + s.val, 0) || 1;

    let barsHtml = "";
    let runningOffset = 0;

    activeSteps.forEach((step) => {
      const pct = Math.max(2, Math.min(100, (step.val / total) * 100));
      const offsetPct = Math.min(98, (runningOffset / total) * 100);
      runningOffset += step.val;

      barsHtml += `
        <div class="waterfall-row">
          <div class="waterfall-label">
            <span class="legend-dot" style="background:${step.color};"></span>
            <span>${step.label}</span>
          </div>
          <div class="waterfall-track">
            <div class="waterfall-bar" style="left: ${offsetPct.toFixed(1)}%; width: ${pct.toFixed(1)}%; background: ${step.color};" title="${step.label}: ${step.val.toFixed(2)} ms">
              <span class="waterfall-bar-label">${step.val.toFixed(2)}ms</span>
            </div>
          </div>
          <div class="waterfall-val">${step.val.toFixed(2)} ms</div>
        </div>
      `;
    });

    container.innerHTML = `
      <div class="waterfall-container">
        <div class="waterfall-header">
          <span>Retrieval Pipeline Stage</span>
          <span class="waterfall-total">Total Latency: <strong>${total.toFixed(2)} ms</strong></span>
        </div>
        <div class="waterfall-rows">
          ${barsHtml}
        </div>
      </div>
    `;
  },

  /**
   * Render comparative IR evaluation metrics bars.
   * @param {HTMLElement} container
   * @param {Array} evalData - Array of pipeline evaluation objects from benchmark.json
   */
  renderBenchmarkComparison(container, evalData) {
    if (!container) return;
    if (!evalData || evalData.length === 0) {
      container.innerHTML = `<div class="chart-empty">Run evaluation suite to populate comparative IR quality benchmarks.</div>`;
      return;
    }

    const metrics = [
      { key: "mrr_at_k", label: "MRR@10", max: 1.0 },
      { key: "recall_at_k", label: "Recall@10", max: 1.0 },
      { key: "ndcg_at_k", label: "nDCG@10", max: 1.0 },
      { key: "precision_at_k", label: "Precision@10", max: 1.0 },
    ];

    const pipelineColors = {
      tfidf: "#64748B",
      semantic: "#c7a5ff",
      hybrid: "#79a2f6",
      hybrid_rerank: "#009bdc",
    };

    const pipelineNames = {
      tfidf: "TF-IDF Baseline",
      semantic: "Dense Semantic",
      hybrid: "Hybrid (Dense + BM25)",
      hybrid_rerank: "Hybrid + Reranker",
    };

    let metricsHtml = "";

    metrics.forEach((metric) => {
      let barsHtml = "";
      evalData.forEach((pipe) => {
        const val = pipe[metric.key] || 0;
        const pct = (val / metric.max) * 100;
        const color = pipelineColors[pipe.pipeline_name] || "#5B8CFF";
        const name = pipelineNames[pipe.pipeline_name] || pipe.pipeline_name;

        barsHtml += `
          <div class="bench-bar-group">
            <div class="bench-bar-track">
              <div class="bench-bar-fill" style="height: ${pct.toFixed(1)}%; background: ${color};" title="${name}: ${val.toFixed(4)}">
                <span class="bench-bar-val">${val.toFixed(3)}</span>
              </div>
            </div>
            <div class="bench-bar-name">${pipe.pipeline_name.replace("hybrid_rerank", "rerank")}</div>
          </div>
        `;
      });

      metricsHtml += `
        <div class="bench-metric-card">
          <div class="bench-metric-header">
            <h4>${metric.label}</h4>
            <span class="bench-metric-sub">Ranked Retrieval Calibration</span>
          </div>
          <div class="bench-chart-area">
            ${barsHtml}
          </div>
        </div>
      `;
    });

    container.innerHTML = `
      <div class="bench-comparison-grid">
        ${metricsHtml}
      </div>
    `;
  },

  /**
   * Render concurrency stress throughput & latency chart.
   * @param {HTMLElement} container
   * @param {Array} loadData - Array of concurrency benchmark results
   */
  renderConcurrencyThroughput(container, loadData) {
    if (!container) return;
    if (!loadData || loadData.length === 0) {
      container.innerHTML = `<div class="chart-empty">Run load benchmark script to populate concurrency stress results.</div>`;
      return;
    }

    let rowsHtml = "";
    loadData.forEach((r) => {
      const rps = r.requests_per_second.toFixed(1);
      const p50 = r.p50_ms.toFixed(1);
      const p95 = r.p95_ms.toFixed(1);
      const p99 = r.p99_ms.toFixed(1);
      const errRate = (r.error_rate * 100).toFixed(2);
      const errClass = r.errors === 0 ? "badge-success" : "badge-danger";

      rowsHtml += `
        <tr>
          <td><strong class="font-mono text-accent">${r.concurrency} workers</strong></td>
          <td class="text-right font-mono">${r.total_requests.toLocaleString()}</td>
          <td class="text-right font-mono"><strong style="color:var(--accent-primary);">${rps}</strong></td>
          <td class="text-right font-mono">${p50} ms</td>
          <td class="text-right font-mono">${p95} ms</td>
          <td class="text-right font-mono">${p99} ms</td>
          <td class="text-center"><span class="badge ${errClass}">${errRate}%</span></td>
        </tr>
      `;
    });

    container.innerHTML = `
      <div class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Concurrency</th>
              <th class="text-right">Total Requests</th>
              <th class="text-right">Throughput (RPS)</th>
              <th class="text-right">P50 Latency</th>
              <th class="text-right">P95 Latency</th>
              <th class="text-right">P99 Latency</th>
              <th class="text-center">SLA Error Rate</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
    `;
  },
};

window.Charts = Charts;
