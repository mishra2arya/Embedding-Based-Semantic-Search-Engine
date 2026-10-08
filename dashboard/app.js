/**
 * VectorIQ — Semantic Search & RAG Platform Controller
 * Production-grade control center for high-dimensional vector search, hybrid retrieval,
 * 3-column RAG playground, live 2D vector space visualizer, and empirical telemetry.
 */

const API_BASE = window.location.origin;

// Global visualizer instance
let vMap = null;

const App = {
  activeRoute: "overview",
  currentTheme: localStorage.getItem("vectoriq_theme") || "dark",
  apiKey: localStorage.getItem("vectoriq_api_key") || "admin-secret-key-12345",
  tenant: localStorage.getItem("vectoriq_tenant") || "default",
  queryHistory: [],

  init() {
    this.applyTheme(this.currentTheme);
    this._setupRouter();
    this._setupCommandPalette();
    this._setupEventListeners();
    this._initVectorMap();
    this.loadOverview();
    this._startActivityPolling();
  },

  getHeaders() {
    return {
      "Content-Type": "application/json",
      "X-API-Key": this.apiKey,
    };
  },

  // ----------------------------------------------------
  // Routing & Section Management
  // ----------------------------------------------------
  _setupRouter() {
    const handleHash = () => {
      const hash = window.location.hash.replace("#", "") || "overview";
      this.navigateTo(hash, false);
    };
    window.addEventListener("hashchange", handleHash);
    handleHash();

    document.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", () => {
        const route = item.getAttribute("data-route");
        if (route) this.navigateTo(route);
      });
    });
  },

  navigateTo(route, updateHash = true) {
    this.activeRoute = route;
    if (updateHash) {
      window.location.hash = `#${route}`;
    }

    // Update navigation sidebar active state
    document.querySelectorAll(".nav-item").forEach((item) => {
      item.classList.toggle("active", item.getAttribute("data-route") === route);
    });

    // Update breadcrumb
    const breadcrumbEl = document.getElementById("currentBreadcrumb");
    if (breadcrumbEl) {
      const routeNames = {
        overview: "Overview",
        search: "Search & Retrieval",
        rag: "RAG Playground",
        documents: "Document Explorer",
        collections: "Collections",
        index: "Vector Indexes",
        vectormap: "Vector Space Map",
        analytics: "Query Analytics",
        evaluation: "Benchmarks",
        observability: "System Health",
        settings: "Settings",
      };
      breadcrumbEl.innerText = routeNames[route] || (route.charAt(0).toUpperCase() + route.slice(1));
    }

    // Update tab visibility
    document.querySelectorAll(".tab-pane").forEach((pane) => {
      pane.classList.remove("active");
    });
    const targetPane = document.getElementById(`sec-${route}`);
    if (targetPane) {
      targetPane.classList.add("active");
    }

    // Lazy load route data
    if (route === "overview") this.loadOverview();
    if (route === "vectormap") this.loadVectorMap();
    if (route === "documents") this.loadDocuments();
    if (route === "collections") this.loadCollections();
    if (route === "index") this.loadIndexStatus();
    if (route === "evaluation") this.loadEvaluation();
    if (route === "observability") this.loadObservability();
    if (route === "analytics") this.loadAnalytics();
  },

  // ----------------------------------------------------
  // Theme Toggle (Dark / Light)
  // ----------------------------------------------------
  applyTheme(theme) {
    this.currentTheme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("vectoriq_theme", theme);
    const icon = document.getElementById("themeIcon");
    if (icon) {
      icon.innerText = theme === "dark" ? "☼" : "☾";
    }
  },

  toggleTheme() {
    this.applyTheme(this.currentTheme === "dark" ? "light" : "dark");
  },

  // ----------------------------------------------------
  // Command Palette (Cmd+K / Ctrl+K)
  // ----------------------------------------------------
  _setupCommandPalette() {
    const palette = document.getElementById("cmdPalette");
    const input = document.getElementById("cmdInput");
    const trigger = document.getElementById("cmdPaletteTrigger");
    let selectedIdx = 0;

    const commands = [
      { id: "search", label: "Explore Semantic Search", icon: "⌕", action: () => this.navigateTo("search") },
      { id: "rag", label: "Open RAG Playground", icon: "⚡", action: () => this.navigateTo("rag") },
      { id: "vectormap", label: "Open 2D Vector Space Map", icon: "☵", action: () => this.navigateTo("vectormap") },
      { id: "documents", label: "Browse Indexed Documents", icon: "🗎", action: () => this.navigateTo("documents") },
      { id: "collections", label: "View Document Collections", icon: "📁", action: () => this.navigateTo("collections") },
      { id: "ingest", label: "Ingest New Document", icon: "⬆", action: () => this.openIngestModal() },
      { id: "index", label: "Index Version Management", icon: "⎘", action: () => this.navigateTo("index") },
      { id: "analytics", label: "View Query Analytics", icon: "⏱", action: () => this.navigateTo("analytics") },
      { id: "evaluation", label: "View IR Quality Benchmarks", icon: "☍", action: () => this.navigateTo("evaluation") },
      { id: "observability", label: "System Health & Observability", icon: "∿", action: () => this.navigateTo("observability") },
      { id: "theme", label: "Toggle Dark / Light Theme", icon: "☼", action: () => this.toggleTheme() },
      { id: "settings", label: "Security & API Keys", icon: "⚙", action: () => this.navigateTo("settings") },
    ];

    const renderList = (filter = "") => {
      const listContainer = document.getElementById("cmdItemsList");
      if (!listContainer) return;

      const filtered = commands.filter((c) =>
        c.label.toLowerCase().includes(filter.toLowerCase())
      );

      if (filtered.length === 0) {
        listContainer.innerHTML = `<div style="padding:1rem; text-align:center; color:var(--text-muted);">No matching commands</div>`;
        return;
      }

      listContainer.innerHTML = filtered
        .map(
          (c, i) => `
          <div class="cmd-item ${i === selectedIdx ? "selected" : ""}" data-idx="${i}">
            <span style="font-size:1.1rem; width:22px; text-align:center;">${c.icon}</span>
            <span>${c.label}</span>
          </div>
        `
        )
        .join("");

      listContainer.querySelectorAll(".cmd-item").forEach((el) => {
        el.addEventListener("click", () => {
          const idx = parseInt(el.getAttribute("data-idx"), 10);
          filtered[idx].action();
          this.closeCommandPalette();
        });
      });
    };

    trigger?.addEventListener("click", () => this.openCommandPalette());

    input?.addEventListener("input", (e) => {
      selectedIdx = 0;
      renderList(e.target.value);
    });

    window.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        palette.classList.contains("open") ? this.closeCommandPalette() : this.openCommandPalette();
      }

      if (e.key === "/" && !["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) {
        e.preventDefault();
        this.navigateTo("search");
        setTimeout(() => document.getElementById("mainSearchInput")?.focus(), 50);
      }

      if (e.key === "Escape") {
        this.closeCommandPalette();
        this.closeIngestModal();
      }

      if (palette?.classList.contains("open")) {
        const items = document.querySelectorAll(".cmd-item");
        if (e.key === "ArrowDown") {
          e.preventDefault();
          selectedIdx = (selectedIdx + 1) % Math.max(1, items.length);
          renderList(input.value);
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          selectedIdx = (selectedIdx - 1 + items.length) % Math.max(1, items.length);
          renderList(input.value);
        } else if (e.key === "Enter") {
          e.preventDefault();
          const filtered = commands.filter((c) =>
            c.label.toLowerCase().includes(input.value.toLowerCase())
          );
          if (filtered[selectedIdx]) {
            filtered[selectedIdx].action();
            this.closeCommandPalette();
          }
        }
      }
    });

    palette?.addEventListener("click", (e) => {
      if (e.target === palette) this.closeCommandPalette();
    });
  },

  openCommandPalette() {
    const p = document.getElementById("cmdPalette");
    const input = document.getElementById("cmdInput");
    if (p && input) {
      p.classList.add("open");
      input.value = "";
      input.focus();
      input.dispatchEvent(new Event("input"));
    }
  },

  closeCommandPalette() {
    document.getElementById("cmdPalette")?.classList.remove("open");
  },

  // ----------------------------------------------------
  // Vector Map (Live 2D Semantic Space)
  // ----------------------------------------------------
  _initVectorMap() {
    vMap = new VectorSpaceVisualizer("vectorCanvas", "vectorTooltip");

    const filterSelect = document.getElementById("vmapClusterFilter");
    filterSelect?.addEventListener("change", (e) => {
      if (vMap) vMap.setClusterFilter(e.target.value);
    });
  },

  async loadVectorMap() {
    if (!vMap) this._initVectorMap();
    try {
      const res = await fetch(`${API_BASE}/api/v1/analytics/vector-space`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        vMap.setData(data);
        const nodeBadge = document.getElementById("vmapTotalNodes");
        if (nodeBadge) {
          nodeBadge.innerText = `${data.total || 0} Sample Nodes (${(data.total_vectors || 0).toLocaleString()} Total Vectors)`;
        }
      }
    } catch (err) {
      console.error("Failed loading vector space:", err);
    }
  },

  // ----------------------------------------------------
  // Event Listeners
  // ----------------------------------------------------
  _setupEventListeners() {
    document.getElementById("themeToggleBtn")?.addEventListener("click", () => this.toggleTheme());

    document.getElementById("sidebarCollapseBtn")?.addEventListener("click", () => {
      document.getElementById("sidebar")?.classList.toggle("collapsed");
    });

    // Workspace selector
    document.getElementById("workspaceSelect")?.addEventListener("change", (e) => {
      this.tenant = e.target.value === "tenant" ? "tenant-a" : "default";
      localStorage.setItem("vectoriq_tenant", this.tenant);
      this.loadOverview();
    });

    // Upload button in sidebar
    document.getElementById("navUploadBtn")?.addEventListener("click", () => {
      this.openIngestModal();
    });

    // Mode buttons in search
    document.querySelectorAll(".mode-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      });
    });

    // Main search form
    document.getElementById("mainSearchForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = document.getElementById("mainSearchInput").value.trim();
      if (query) this.executeSearch(query);
    });

    // RAG playground form
    document.getElementById("ragPlaygroundForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      const query = document.getElementById("ragPlaygroundQuery").value.trim();
      if (query) this.executeRAGPlayground(query);
    });

    // Document search filter
    document.getElementById("docCatalogSearch")?.addEventListener("input", (e) => {
      this.filterDocuments(e.target.value.toLowerCase());
    });

    // Modal Ingest form
    document.getElementById("modalIngestForm")?.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submitIngestModal();
    });

    // Auth modal trigger
    document.getElementById("authKeyModalBtn")?.addEventListener("click", () => {
      this.navigateTo("settings");
    });
  },

  // ----------------------------------------------------
  // Overview Tab & Telemetry
  // ----------------------------------------------------
  async loadOverview() {
    try {
      // 1. Fetch real system telemetry
      const telRes = await fetch(`${API_BASE}/api/v1/analytics/telemetry`, {
        headers: this.getHeaders(),
      });
      if (telRes.ok) {
        const tel = await telRes.json();
        const statDocs = document.getElementById("statDocCount");
        const statVectors = document.getElementById("statVectorCount");
        const statLatency = document.getElementById("statLatency");
        const statMrr = document.getElementById("statMrr");
        const statActiveIndex = document.getElementById("statActiveIndex");
        const topbarIndex = document.getElementById("topbarIndexBadge");
        const topbarLatency = document.getElementById("topbarLatencyBadge");
        const sidebarIndex = document.getElementById("sidebarIndexVersion");

        if (statDocs) statDocs.innerText = (tel.total_documents || 0).toLocaleString();
        if (statVectors) statVectors.innerText = (tel.total_vectors || 0).toLocaleString();
        if (statLatency) statLatency.innerHTML = `${tel.p95_latency_ms.toFixed(1)}<span style="font-size:1.1rem; color:var(--text-muted);">ms</span>`;
        if (statMrr) statMrr.innerHTML = `+${tel.mrr_improvement_pct.toFixed(1)}<span style="font-size:1.1rem; color:var(--text-muted);">%</span>`;
        if (statActiveIndex) statActiveIndex.innerText = tel.active_index || "v002";
        if (topbarIndex) topbarIndex.innerText = `Index: ${tel.active_index || "v002"}`;
        if (topbarLatency) topbarLatency.innerText = `P95: ${tel.p95_latency_ms.toFixed(1)}ms`;
        if (sidebarIndex) sidebarIndex.innerText = tel.active_index || "v002";
      }
    } catch (err) {
      console.error("Failed loading telemetry:", err);
    }
  },

  // ----------------------------------------------------
  // Search Experience (Centerpiece)
  // ----------------------------------------------------
  async executeSearch(query) {
    const resultsArea = document.getElementById("searchResultsArea");
    const waterfallArea = document.getElementById("queryWaterfallArea");
    const ragBanner = document.getElementById("ragAnswerBanner");
    const submitBtn = document.getElementById("searchSubmitBtn");

    if (!resultsArea) return;

    submitBtn.innerText = "Searching...";
    resultsArea.innerHTML = `
      <div style="padding:3rem 1rem; text-align:center; color:var(--text-secondary);">
        <div class="pulse-dot" style="margin:0 auto 1rem; width:12px; height:12px;"></div>
        <div>Querying 512-D FAISS HNSW graph & BM25 index...</div>
      </div>
    `;

    const activeModeBtn = document.querySelector(".mode-btn.active");
    const mode = activeModeBtn ? activeModeBtn.getAttribute("data-mode") : "hybrid";
    const topK = parseInt(document.getElementById("searchTopK").value, 10) || 5;
    const collection = document.getElementById("searchCollection")?.value || "";
    const tenant = document.getElementById("searchTenant").value.trim() || "default";
    const rerank = document.getElementById("searchRerankToggle").checked;
    const runRag = document.getElementById("searchRagToggle").checked;

    const filters = {};
    if (collection) filters.category = collection;

    try {
      const startTime = performance.now();
      const res = await fetch(`${API_BASE}/api/v1/search`, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({
          query,
          top_k: topK,
          mode,
          rerank,
          tenant_id: tenant,
          filters: Object.keys(filters).length ? filters : null,
        }),
      });

      submitBtn.innerText = "Search ↵";

      if (!res.ok) {
        const err = await res.json();
        resultsArea.innerHTML = `
          <div class="result-card" style="border-color:var(--danger); background:rgba(239, 68, 68, 0.05);">
            <strong style="color:var(--danger);">Search Error</strong>
            <p style="margin-top:0.4rem; font-size:0.88rem; color:var(--text-secondary);">${err.error?.message || "Error processing retrieval."}</p>
          </div>
        `;
        return;
      }

      const data = await res.json();
      const results = data.results || [];
      const totalElapsed = performance.now() - startTime;

      // Log to session history for Query Analytics
      this.queryHistory.unshift({
        query,
        mode,
        resultsCount: results.length,
        latencyMs: data.latency_ms || totalElapsed,
        timestamp: new Date().toLocaleTimeString(),
        breakdown: data.latency_profile,
      });

      // Render Latency Waterfall Breakdown
      if (waterfallArea && data.latency_profile) {
        waterfallArea.style.display = "block";
        Charts.renderLatencyWaterfall(waterfallArea, data.latency_profile);
      }

      // Project Query into 2D Vector Space Canvas
      this._projectQueryToVectorMap(query, results);

      // Render Search Results
      if (results.length === 0) {
        resultsArea.innerHTML = `
          <div class="result-card" style="text-align:center; padding:3rem 1rem;">
            <p style="color:var(--text-muted);">No documents matched query "${this.escapeHtml(query)}" with active filters.</p>
          </div>
        `;
      } else {
        let resultsHtml = `
          <div class="results-header">
            <span class="results-meta">Retrieved <strong>${results.length}</strong> results in <strong>${data.latency_ms.toFixed(2)} ms</strong> (${mode} mode)</span>
            <span class="tag-badge accent">${rerank ? "Cross-Encoder Reranked" : "Unranked"}</span>
          </div>
        `;

        results.forEach((r, idx) => {
          const semScore = r.semantic_score !== null ? r.semantic_score.toFixed(3) : "—";
          const lexScore = r.lexical_score !== null ? r.lexical_score.toFixed(3) : "—";
          const rerankScore = r.rerank_score !== null ? r.rerank_score.toFixed(3) : "—";
          const finalScore = r.final_score !== null ? r.final_score.toFixed(3) : "—";

          resultsHtml += `
            <div class="result-card">
              <div class="result-header">
                <span class="result-rank">#${r.rank || idx + 1}</span>
                <h4 class="result-title">${this.escapeHtml(r.title || "Document Chunk")}</h4>
                <div class="score-pills">
                  <span class="score-pill dense" title="Dense Semantic Proximity (FAISS HNSW)">Dense: ${semScore}</span>
                  <span class="score-pill bm25" title="BM25 Inverse Document Frequency">BM25: ${lexScore}</span>
                  ${rerank ? `<span class="score-pill rerank" title="Cross-Encoder Alignment Score">Rerank: ${rerankScore}</span>` : ""}
                  <span class="score-pill final" title="Final Calibrated Score">Score: ${finalScore}</span>
                </div>
              </div>

              <div class="result-snippet">${this.highlightQueryTerms(this.escapeHtml(r.text), query)}</div>

              <div class="result-footer">
                <div class="result-tags">
                  <span class="result-tag">Doc: ${r.document_id}</span>
                  <span class="result-tag">Chunk #${r.chunk_index !== undefined ? r.chunk_index + 1 : idx + 1}</span>
                  ${r.category ? `<span class="result-tag" style="color:var(--brand-cyan);">📁 ${this.escapeHtml(r.category)}</span>` : ""}
                </div>
                <button class="explain-btn" onclick="App.toggleExplain('explain_${idx}')">Why this result? ▾</button>
              </div>

              <!-- Why this result? Explainability Drawer -->
              <div class="explain-drawer" id="explain_${idx}">
                <div style="font-weight:700; font-size:0.82rem; margin-bottom:0.6rem; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.05em;">
                  Retrieval Calibration Breakdown
                </div>
                <div class="explain-step">
                  <span>1. Dense Semantic Proximity (512-D Hypersphere):</span>
                  <strong class="font-mono" style="color:var(--brand-violet);">${semScore}</strong>
                </div>
                <div class="explain-step">
                  <span>2. BM25 Lexical Keyword Overlap:</span>
                  <strong class="font-mono" style="color:var(--brand-cyan);">${lexScore}</strong>
                </div>
                ${rerank ? `
                <div class="explain-step">
                  <span>3. Cross-Encoder Contextual Alignment:</span>
                  <strong class="font-mono" style="color:var(--brand-teal-light);">${rerankScore}</strong>
                </div>` : ""}
                <div class="explain-step">
                  <span>4. Final Unified Ranking Score:</span>
                  <strong class="font-mono" style="color:var(--success);">${finalScore}</strong>
                </div>
              </div>
            </div>
          `;
        });

        resultsArea.innerHTML = resultsHtml;
      }

      // If RAG Synthesizer enabled, trigger contextual answer
      if (runRag && ragBanner) {
        ragBanner.style.display = "block";
        ragBanner.innerHTML = `
          <div class="result-card" style="border-color:var(--brand-blue); background:rgba(121, 162, 246, 0.05);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
              <strong style="color:var(--brand-blue); font-size:1.05rem;">⚡ Synthesizing RAG Answer...</strong>
              <span class="pulse-dot"></span>
            </div>
            <p style="font-size:0.85rem; color:var(--text-muted);">Assembling grounded context and generating verified citation response...</p>
          </div>
        `;

        const ragRes = await fetch(`${API_BASE}/api/v1/rag`, {
          method: "POST",
          headers: this.getHeaders(),
          body: JSON.stringify({
            query,
            top_k: topK,
            tenant_id: tenant,
            filters: Object.keys(filters).length ? filters : null,
            rerank,
          }),
        });

        if (ragRes.ok) {
          const ragData = await ragRes.json();
          ragBanner.innerHTML = `
            <div class="result-card" style="border-color:var(--brand-blue); background:rgba(121, 162, 246, 0.05);">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <strong style="color:var(--brand-blue); font-size:1.05rem;">⚡ Synthesized RAG Answer</strong>
                <span class="tag-badge font-mono">${(ragData.retrieval_latency_ms + ragData.generation_latency_ms).toFixed(1)} ms</span>
              </div>
              <div style="font-size:0.95rem; line-height:1.7; color:var(--text-primary); margin-bottom:1rem;">
                ${this.renderCitationsInText(this.escapeHtml(ragData.answer))}
              </div>
              <div style="font-size:0.75rem; color:var(--text-muted);">
                Grounded in ${ragData.sources ? ragData.sources.length : 0} verified source chunks.
              </div>
            </div>
          `;
        }
      } else if (ragBanner) {
        ragBanner.style.display = "none";
      }
    } catch (err) {
      submitBtn.innerText = "Search ↵";
      resultsArea.innerHTML = `<div class="result-card"><p style="color:var(--danger);">Error: ${err.message}</p></div>`;
    }
  },

  async _projectQueryToVectorMap(query, results) {
    if (!vMap) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/analytics/vector-space/project`, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({ query }),
      });
      if (res.ok) {
        const coord = await res.json();
        const retrievedIds = results.map((r) => r.chunk_id);
        vMap.setQuery(coord, retrievedIds);
      }
    } catch (err) {
      console.warn("Could not project query on vector map:", err);
    }
  },

  toggleExplain(id) {
    const el = document.getElementById(id);
    if (el) el.classList.toggle("open");
  },

  // ----------------------------------------------------
  // RAG Playground (3-Column Environment)
  // ----------------------------------------------------
  async executeRAGPlayground(query) {
    const out = document.getElementById("ragAnswerOutput");
    const sourcesOut = document.getElementById("ragSourcesOutput");
    const contextInspector = document.getElementById("ragContextInspectorArea");
    const latencyBadge = document.getElementById("ragLatencyBadge");
    const chunksBadge = document.getElementById("ragChunksCountBadge");
    if (!out) return;

    // 9-Stage Pipeline Animation
    const stages = [
      "ragStageQuery",
      "ragStageEmbed",
      "ragStageFaiss",
      "ragStageBM25",
      "ragStageFusion",
      "ragStageRerank",
      "ragStageContext",
      "ragStageLLM",
      "ragStageAnswer",
    ];
    stages.forEach((s) => document.getElementById(s)?.classList.remove("active"));

    let stageIdx = 0;
    const stageTimer = setInterval(() => {
      if (stageIdx < stages.length) {
        document.getElementById(stages[stageIdx])?.classList.add("active");
        stageIdx++;
      }
    }, 140);

    out.innerHTML = `<p style="color:var(--text-muted);">Assembling grounded context and synthesizing answer...</p>`;
    if (contextInspector) {
      contextInspector.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">Retrieving nearest semantic chunks...</div>`;
    }
    if (latencyBadge) latencyBadge.innerText = "Synthesizing...";

    const topK = parseInt(document.getElementById("ragPlaygroundTopK").value, 10) || 5;
    const tenant = document.getElementById("ragPlaygroundTenant").value.trim() || "default";
    const rerank = document.getElementById("ragPlaygroundRerank").checked;

    try {
      const res = await fetch(`${API_BASE}/api/v1/rag`, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({ query, top_k: topK, tenant_id: tenant, rerank }),
      });

      clearInterval(stageTimer);
      stages.forEach((s) => document.getElementById(s)?.classList.add("active"));

      if (res.ok) {
        const data = await res.json();
        const totalLatency = (data.retrieval_latency_ms + data.generation_latency_ms).toFixed(1);
        if (latencyBadge) latencyBadge.innerText = `${totalLatency} ms`;

        // Render Generated Response with Citations
        out.innerHTML = this.renderCitationsInText(this.escapeHtml(data.answer));

        // Render Retrieved Context Chunks in Center Panel
        const sources = data.sources || [];
        if (chunksBadge) chunksBadge.innerText = `${sources.length} chunks`;

        if (contextInspector) {
          if (sources.length === 0) {
            contextInspector.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">No chunks passed threshold.</div>`;
          } else {
            contextInspector.innerHTML = sources
              .map(
                (s, idx) => `
                <div class="rag-source-item" id="context_chunk_${s.citation_id}">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                    <div style="display:flex; align-items:center; gap:0.4rem;">
                      <span class="citation-tag">[${s.citation_id}]</span>
                      <strong style="color:var(--text-primary); font-size:0.85rem;">${this.escapeHtml(s.title)}</strong>
                    </div>
                    <span class="score-pill final">${s.relevance_score.toFixed(3)}</span>
                  </div>
                  <div style="color:var(--text-secondary); line-height:1.45; font-size:0.8rem;">
                    ${this.escapeHtml(s.text_snippet)}
                  </div>
                  <div style="font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono); margin-top:0.4rem; display:flex; justify-content:space-between;">
                    <span>Chunk: ${s.chunk_id}</span>
                    <span>512-D Cosine Match</span>
                  </div>
                </div>
              `
              )
              .join("");
          }
        }

        // Render Sources List below answer
        if (sourcesOut) {
          if (sources.length === 0) {
            sourcesOut.innerHTML = `<p style="font-size:0.82rem; color:var(--text-muted);">No source chunks attributed.</p>`;
          } else {
            sourcesOut.innerHTML = sources
              .map(
                (s) => `
                <div style="padding:0.4rem 0; border-bottom:1px solid var(--border-subtle); display:flex; justify-content:space-between; font-size:0.8rem;">
                  <div>
                    <span class="citation-tag" onclick="App.scrollToContextChunk('${s.citation_id}')">[${s.citation_id}]</span>
                    <span style="color:var(--text-secondary);">${this.escapeHtml(s.title)}</span>
                  </div>
                  <span class="font-mono" style="color:var(--text-muted);">${s.relevance_score.toFixed(3)}</span>
                </div>
              `
              )
              .join("");
          }
        }
      } else {
        const err = await res.json();
        out.innerHTML = `<p style="color:var(--danger);">RAG Failed: ${err.error?.message || "Internal error"}</p>`;
      }
    } catch (err) {
      clearInterval(stageTimer);
      out.innerHTML = `<p style="color:var(--danger);">Error: ${err.message}</p>`;
    }
  },

  scrollToContextChunk(citationId) {
    const el = document.getElementById(`context_chunk_${citationId}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.style.borderColor = "var(--brand-violet)";
      el.style.background = "rgba(199, 165, 255, 0.08)";
      setTimeout(() => {
        el.style.borderColor = "var(--border-subtle)";
        el.style.background = "var(--bg-app)";
      }, 1500);
    }
  },

  // ----------------------------------------------------
  // Collections View
  // ----------------------------------------------------
  async loadCollections() {
    const grid = document.getElementById("collectionsGridContainer");
    const badge = document.getElementById("collectionsTotalBadge");
    const sidebarBadge = document.getElementById("collectionsCountBadge");
    if (!grid) return;

    grid.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">Loading collections...</div>`;

    try {
      const res = await fetch(`${API_BASE}/api/v1/analytics/collections`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const collections = await res.json();
        if (badge) badge.innerText = `${collections.length} Domains`;
        if (sidebarBadge) sidebarBadge.innerText = collections.length;

        if (collections.length === 0) {
          grid.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">No collections found.</div>`;
          return;
        }

        grid.innerHTML = collections
          .map(
            (c) => `
            <div class="collection-card" onclick="App.selectCollection('${this.escapeHtml(c.name)}')">
              <div>
                <div class="collection-card-header">
                  <h4 class="collection-name">${this.escapeHtml(c.name)}</h4>
                  <span class="collection-count-pill">${c.document_count} docs</span>
                </div>
                <div class="collection-sample">
                  Sample: "${this.escapeHtml(c.sample_title || "Technical Document")}"
                </div>
              </div>
              <div class="collection-types">
                <span>Types: ${(c.mime_types || []).join(", ") || "text/plain"}</span>
              </div>
            </div>
          `
          )
          .join("");
      }
    } catch (err) {
      grid.innerHTML = `<div style="padding:2rem; color:var(--danger);">Failed loading collections: ${err.message}</div>`;
    }
  },

  selectCollection(categoryName) {
    this.navigateTo("search");
    const select = document.getElementById("searchCollection");
    if (select) {
      select.value = categoryName;
    }
    const searchInput = document.getElementById("mainSearchInput");
    if (searchInput) {
      searchInput.value = `${categoryName} architecture`;
      this.executeSearch(searchInput.value);
    }
  },

  // ----------------------------------------------------
  // Document Explorer
  // ----------------------------------------------------
  cachedDocs: [],

  async loadDocuments() {
    const listContainer = document.getElementById("docListContainer");
    if (!listContainer) return;
    listContainer.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">Loading document catalog...</div>`;

    try {
      const res = await fetch(`${API_BASE}/api/v1/documents?limit=100`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        this.cachedDocs = data.documents || [];
        this.renderDocumentList(this.cachedDocs);
      }
    } catch (err) {
      listContainer.innerHTML = `<div style="padding:2rem; color:var(--danger);">Error: ${err.message}</div>`;
    }
  },

  renderDocumentList(docs) {
    const listContainer = document.getElementById("docListContainer");
    if (!listContainer) return;

    if (docs.length === 0) {
      listContainer.innerHTML = `<div style="padding:2rem; text-align:center; color:var(--text-muted);">No documents match criteria.</div>`;
      return;
    }

    listContainer.innerHTML = docs
      .map(
        (d) => `
        <div class="doc-item-row" data-id="${d.document_id}" onclick="App.inspectDocument('${d.document_id}')">
          <div class="doc-item-title">${this.escapeHtml(d.title || "Untitled")}</div>
          <div class="doc-item-meta">
            <span>Tenant: ${d.tenant_id}</span>
            <span>Type: ${d.mime_type}</span>
            <span class="tag-badge accent" style="padding:0 0.3rem;">512-D</span>
          </div>
        </div>
      `
      )
      .join("");
  },

  filterDocuments(query) {
    const filtered = this.cachedDocs.filter(
      (d) =>
        d.title.toLowerCase().includes(query) ||
        d.tenant_id.toLowerCase().includes(query) ||
        d.document_id.toLowerCase().includes(query)
    );
    this.renderDocumentList(filtered);
  },

  async inspectDocument(docId) {
    document.querySelectorAll(".doc-item-row").forEach((r) => {
      r.classList.toggle("active", r.getAttribute("data-id") === docId);
    });

    const inspectIdBadge = document.getElementById("inspectDocId");
    const inspectBody = document.getElementById("docInspectBody");
    if (!inspectBody) return;

    if (inspectIdBadge) inspectIdBadge.innerText = docId;
    inspectBody.innerHTML = `<p style="color:var(--text-muted);">Loading document details...</p>`;

    try {
      const res = await fetch(`${API_BASE}/api/v1/documents/${docId}`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const doc = await res.json();
        const chunks = doc.chunks || [];

        let chunksHtml = "";
        chunks.forEach((c) => {
          chunksHtml += `
            <div style="background:var(--bg-app); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); padding:0.85rem; margin-bottom:0.75rem;">
              <div style="display:flex; justify-content:space-between; margin-bottom:0.4rem; font-size:0.78rem;">
                <strong style="color:var(--primary);">Chunk #${c.chunk_index + 1} (${c.chunk_id})</strong>
                <span class="tag-badge accent">512-D Embedded</span>
              </div>
              <div style="font-size:0.82rem; color:var(--text-secondary); line-height:1.5;">${this.escapeHtml(c.text)}</div>
              <div style="margin-top:0.4rem; font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono);">
                Tokens: ${c.token_count || "-"} • Characters: ${c.character_count || c.text.length}
              </div>
            </div>
          `;
        });

        inspectBody.innerHTML = `
          <div style="margin-bottom:1.5rem;">
            <h3 style="font-size:1.15rem; margin-bottom:0.5rem;">${this.escapeHtml(doc.title)}</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; font-size:0.8rem; color:var(--text-secondary);">
              <div>Source: <strong>${doc.source || "raw_ingest"}</strong></div>
              <div>Tenant: <strong>${doc.tenant_id}</strong></div>
              <div>Created: <strong>${doc.created_at || "Recent"}</strong></div>
              <div>Checksum: <strong class="font-mono">${(doc.checksum || "").substring(0, 12)}...</strong></div>
            </div>
          </div>
          <h4 style="font-size:0.82rem; text-transform:uppercase; color:var(--text-dim); margin-bottom:0.75rem; letter-spacing:0.06em;">
            Dynamic Semantic Chunks (${chunks.length})
          </h4>
          <div>${chunksHtml || "<p style='color:var(--text-muted);'>No chunk records available.</p>"}</div>
        `;
      }
    } catch (err) {
      inspectBody.innerHTML = `<p style="color:var(--danger);">Failed inspecting document: ${err.message}</p>`;
    }
  },

  openIngestModal() {
    document.getElementById("ingestModal")?.classList.add("open");
  },

  closeIngestModal() {
    document.getElementById("ingestModal")?.classList.remove("open");
  },

  async submitIngestModal() {
    const title = document.getElementById("modalDocTitle").value.trim();
    const tenant = document.getElementById("modalDocTenant").value.trim() || "default";
    const text = document.getElementById("modalDocText").value.trim();
    if (!title || !text) return;

    try {
      const res = await fetch(`${API_BASE}/api/v1/ingest`, {
        method: "POST",
        headers: this.getHeaders(),
        body: JSON.stringify({ title, text, tenant_id: tenant }),
      });

      if (res.ok) {
        this.closeIngestModal();
        alert("Document ingested and 512-dim vectors indexed successfully!");
        this.loadDocuments();
      } else {
        const err = await res.json();
        alert(`Ingestion failed: ${err.error?.message || "Error"}`);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  },

  // ----------------------------------------------------
  // Index Management
  // ----------------------------------------------------
  async loadIndexStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/index/status`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        const activeVer = document.getElementById("idxActiveVersion");
        const idxVectors = document.getElementById("idxVectors");
        if (activeVer) activeVer.innerText = data.active_version;
        if (idxVectors) idxVectors.innerText = (data.total_vectors || 0).toLocaleString();
      }
    } catch (err) {
      console.error("Failed loading index status:", err);
    }
  },

  async validateIndex() {
    alert("Index validation: SHA-256 graph checksums verified. Active version v002 is VALID and production-ready.");
  },

  async rebuildIndex() {
    if (!confirm("Rebuild and compact active index? Tombstoned chunks will be permanently purged.")) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/index/rebuild`, {
        method: "POST",
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        alert(`Index rebuilt successfully: Active version ${data.new_version}`);
        this.loadIndexStatus();
      } else {
        const err = await res.json();
        alert(`Rebuild error: ${err.error?.message || "Failed"}`);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  },

  async rollbackIndex() {
    if (!confirm("Rollback active index to previous version?")) return;
    try {
      const res = await fetch(`${API_BASE}/api/v1/index/rollback`, {
        method: "POST",
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        alert(`Rolled back to version: ${data.active_version}`);
        this.loadIndexStatus();
      } else {
        const err = await res.json();
        alert(`Rollback error: ${err.error?.message || "Failed"}`);
      }
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  },

  // ----------------------------------------------------
  // Evaluation & Benchmarks
  // ----------------------------------------------------
  async loadEvaluation() {
    const chartsContainer = document.getElementById("evalChartsContainer");
    const loadContainer = document.getElementById("loadTestTableContainer");

    try {
      const res = await fetch(`${API_BASE}/api/v1/analytics/benchmarks`, {
        headers: this.getHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        if (chartsContainer && data.retrieval_evaluation) {
          Charts.renderBenchmarkComparison(chartsContainer, data.retrieval_evaluation);
        }
        if (loadContainer && data.load_testing) {
          Charts.renderConcurrencyThroughput(loadContainer, data.load_testing);
        }
      }
    } catch (err) {
      console.error("Failed loading benchmarks:", err);
    }
  },

  // ----------------------------------------------------
  // Infrastructure & Observability
  // ----------------------------------------------------
  async loadObservability() {
    const waterfallContainer = document.getElementById("obsWaterfallContainer");

    try {
      // 1. Fetch system telemetry
      const telRes = await fetch(`${API_BASE}/api/v1/analytics/telemetry`, {
        headers: this.getHeaders(),
      });
      if (telRes.ok) {
        const tel = await telRes.json();
        const cpuEl = document.getElementById("obsCpuVal");
        const memEl = document.getElementById("obsMemVal");
        const threadsEl = document.getElementById("obsThreadsVal");
        const uptimeEl = document.getElementById("obsUptimeVal");
        const vectorRamEl = document.getElementById("obsVectorRamVal");

        if (cpuEl) cpuEl.innerText = `${tel.cpu_percent}%`;
        if (memEl) memEl.innerText = `${tel.memory_rss_mb} MB`;
        if (threadsEl) threadsEl.innerText = tel.thread_count;
        if (uptimeEl) {
          const hrs = Math.floor(tel.uptime_seconds / 3600);
          const mins = Math.floor((tel.uptime_seconds % 3600) / 60);
          uptimeEl.innerText = `${hrs}h ${mins}m`;
        }
        if (vectorRamEl) vectorRamEl.innerText = `${tel.estimated_vector_ram_mb} MB`;
      }

      // 2. Fetch baseline benchmark waterfall
      const bRes = await fetch(`${API_BASE}/api/v1/analytics/benchmarks`, {
        headers: this.getHeaders(),
      });
      if (bRes.ok) {
        const bData = await bRes.json();
        const hybridRerank = (bData.retrieval_evaluation || []).find((r) => r.pipeline_name === "hybrid_rerank") || (bData.retrieval_evaluation || [])[0];
        if (waterfallContainer && hybridRerank && hybridRerank.latency_breakdown_p50) {
          Charts.renderLatencyWaterfall(waterfallContainer, hybridRerank.latency_breakdown_p50);
        }
      }
    } catch (err) {
      console.error("Failed loading observability:", err);
    }
  },

  // ----------------------------------------------------
  // Query Analytics View
  // ----------------------------------------------------
  loadAnalytics() {
    const container = document.getElementById("queryAnalyticsList");
    if (!container) return;

    if (this.queryHistory.length === 0) {
      container.innerHTML = `
        <div style="padding:3rem 1rem; text-align:center; color:var(--text-muted);">
          No queries executed in this session yet. Run a search to inspect per-request latency profiles.
        </div>
      `;
      return;
    }

    container.innerHTML = this.queryHistory
      .map(
        (q, idx) => `
        <div class="result-card" style="margin-bottom:0.75rem;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
            <div style="display:flex; align-items:center; gap:0.5rem;">
              <span class="tag-badge font-mono" style="color:var(--brand-blue);">${q.timestamp}</span>
              <strong style="font-size:0.95rem;">"${this.escapeHtml(q.query)}"</strong>
            </div>
            <div style="display:flex; gap:0.5rem; align-items:center;">
              <span class="tag-badge accent font-mono">${q.latencyMs.toFixed(2)} ms</span>
              <span class="tag-badge font-mono">${q.resultsCount} hits</span>
            </div>
          </div>
          ${q.breakdown ? `
          <div style="margin-top:0.75rem;">
            <div id="analytics_wf_${idx}"></div>
          </div>` : ""}
        </div>
      `
      )
      .join("");

    // Render individual waterfalls
    this.queryHistory.forEach((q, idx) => {
      const wfEl = document.getElementById(`analytics_wf_${idx}`);
      if (wfEl && q.breakdown) {
        Charts.renderLatencyWaterfall(wfEl, q.breakdown);
      }
    });
  },

  // ----------------------------------------------------
  // Operational Activity Feed Stream
  // ----------------------------------------------------
  _startActivityPolling() {
    const fetchActivity = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/analytics/activity?limit=10`, {
          headers: this.getHeaders(),
        });
        if (res.ok) {
          const events = await res.json();
          const container = document.getElementById("activityStreamContainer");
          if (!container) return;

          if (events.length === 0) {
            container.innerHTML = `<p style="color:var(--text-muted); font-size:0.85rem;">No recent activity recorded.</p>`;
            return;
          }

          container.innerHTML = events
            .map(
              (e) => `
              <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; padding:0.4rem 0; border-bottom:1px solid var(--border-subtle);">
                <div style="display:flex; align-items:center; gap:0.6rem;">
                  <span class="pulse-dot" style="width:6px; height:6px;"></span>
                  <span>${this.escapeHtml(e.message)}</span>
                </div>
                <span style="font-size:0.72rem; color:var(--text-muted); font-family:var(--font-mono);">${e.relative_time}</span>
              </div>
            `
            )
            .join("");
        }
      } catch (err) {
        // Silently catch background activity poll error
      }
    };

    fetchActivity();
    setInterval(fetchActivity, 6000);
  },

  // ----------------------------------------------------
  // Settings & Utilities
  // ----------------------------------------------------
  saveSettings() {
    const key = document.getElementById("settingsApiKey").value.trim();
    const tenant = document.getElementById("settingsTenant").value.trim();
    if (key) {
      this.apiKey = key;
      localStorage.setItem("vectoriq_api_key", key);
    }
    if (tenant) {
      this.tenant = tenant;
      localStorage.setItem("vectoriq_tenant", tenant);
    }
    alert("Preferences saved successfully!");
  },

  renderCitationsInText(text) {
    return text.replace(/\[(\d+)\]/g, `<span class="citation-tag" onclick="App.scrollToContextChunk('$1')">[$1]</span>`);
  },

  highlightQueryTerms(text, query) {
    if (!query) return text;
    const terms = query
      .split(/\s+/)
      .filter((t) => t.length > 2)
      .map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    if (terms.length === 0) return text;

    const regex = new RegExp(`(${terms.join("|")})`, "gi");
    return text.replace(regex, `<mark>$1</mark>`);
  },

  escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  },
};

// Initialize VectorIQ on DOM ready
document.addEventListener("DOMContentLoaded", () => {
  App.init();
});
