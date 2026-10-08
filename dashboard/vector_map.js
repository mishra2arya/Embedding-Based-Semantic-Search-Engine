/**
 * VectorIQ — Live 2D Semantic Vector Space Visualizer
 * High-performance interactive HTML5 Canvas with real-time query projection,
 * semantic cluster halos, and retrieval proximity filaments.
 */

class VectorSpaceVisualizer {
  constructor(canvasId, tooltipId) {
    this.canvas = document.getElementById(canvasId);
    this.tooltip = document.getElementById(tooltipId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext("2d");
    this.points = [];
    this.clusters = [];
    this.queryPoint = null; // { x, y, text }
    this.retrievedIds = new Set();
    this.activeCluster = "all";

    // Viewport transform
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this.isDragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.hoveredPoint = null;

    // Animation state
    this.animTime = 0;
    this.animFrame = null;

    this.domainColors = {
      ai: "#c7a5ff",
      cloud: "#79a2f6",
      "software engineering": "#009bdc",
      technology: "#009bdc",
      networking: "#008fb5",
      cybersecurity: "#007f88",
      science: "#006c5c",
      finance: "#008fb5",
      documentation: "#79a2f6",
      business: "#007f88",
      default: "#79a2f6",
    };

    this._setupEvents();
    this._handleResize();
    this._startLoop();
  }

  setData(data) {
    if (!data) return;
    this.points = data.points || [];
    this.clusters = data.clusters || [];
    this._render();
  }

  setQuery(queryCoord, retrievedChunkIds = []) {
    this.queryPoint = queryCoord; // { x, y, query }
    this.retrievedIds = new Set(retrievedChunkIds);
    this.animTime = 0;
    this._render();
  }

  setClusterFilter(clusterName) {
    this.activeCluster = clusterName;
    this._render();
  }

  resetView() {
    this.scale = 1.0;
    this.panX = 0;
    this.panY = 0;
    this._render();
  }

  zoom(factor) {
    this.scale = Math.max(0.4, Math.min(4.0, this.scale * factor));
    this._render();
  }

  _handleResize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;

    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
    this._render();
  }

  _worldToScreen(x, y) {
    const cx = this.width / 2 + this.panX;
    const cy = this.height / 2 + this.panY;
    const baseSpan = Math.min(this.width, this.height) * 0.42;
    const sx = cx + (x / 100) * baseSpan * this.scale;
    const sy = cy + (y / 100) * baseSpan * this.scale;
    return { sx, sy };
  }

  _screenToWorld(sx, sy) {
    const cx = this.width / 2 + this.panX;
    const cy = this.height / 2 + this.panY;
    const baseSpan = Math.min(this.width, this.height) * 0.42;
    const x = ((sx - cx) / (baseSpan * this.scale)) * 100;
    const y = ((sy - cy) / (baseSpan * this.scale)) * 100;
    return { x, y };
  }

  _setupEvents() {
    window.addEventListener("resize", () => this._handleResize());

    // Pan drag
    this.canvas.addEventListener("mousedown", (e) => {
      this.isDragging = true;
      this.dragStartX = e.clientX - this.panX;
      this.dragStartY = e.clientY - this.panY;
    });

    window.addEventListener("mousemove", (e) => {
      if (this.isDragging) {
        this.panX = e.clientX - this.dragStartX;
        this.panY = e.clientY - this.dragStartY;
        this._render();
      } else {
        this._handleHover(e);
      }
    });

    window.addEventListener("mouseup", () => {
      this.isDragging = false;
    });

    // Zoom on wheel
    this.canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        this.zoom(delta);
      },
      { passive: false }
    );

    // Double click reset
    this.canvas.addEventListener("dblclick", () => this.resetView());
  }

  _handleHover(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    let nearest = null;
    let minDist = 14; // hover hit radius

    for (const p of this.points) {
      if (this.activeCluster !== "all" && p.domain !== this.activeCluster) continue;
      const { sx, sy } = this._worldToScreen(p.x, p.y);
      const d = Math.hypot(sx - mx, sy - my);
      if (d < minDist) {
        minDist = d;
        nearest = { point: p, sx, sy };
      }
    }

    if (nearest) {
      this.hoveredPoint = nearest.point;
      this._showTooltip(nearest.point, e.clientX, e.clientY);
      this.canvas.style.cursor = "pointer";
    } else {
      this.hoveredPoint = null;
      this._hideTooltip();
      this.canvas.style.cursor = "grab";
    }
  }

  _showTooltip(point, clientX, clientY) {
    if (!this.tooltip) return;
    const color = this.domainColors[point.domain] || "#5B8CFF";
    this.tooltip.innerHTML = `
      <div class="vtooltip-header">
        <span class="legend-dot" style="background:${color};"></span>
        <span class="vtooltip-domain">${point.domain}</span>
        <span class="vtooltip-id font-mono">#${point.faiss_id}</span>
      </div>
      <div class="vtooltip-title">${point.title}</div>
      <div class="vtooltip-snippet">${point.snippet}</div>
      <div class="vtooltip-footer font-mono">Coord: [${point.x.toFixed(1)}, ${point.y.toFixed(1)}]</div>
    `;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = `${clientX + 16}px`;
    this.tooltip.style.top = `${clientY - 12}px`;
  }

  _hideTooltip() {
    if (this.tooltip) this.tooltip.style.display = "none";
  }

  _startLoop() {
    const loop = () => {
      this.animTime += 0.025;
      if (this.queryPoint) {
        this._render();
      }
      this.animFrame = requestAnimationFrame(loop);
    };
    this.animFrame = requestAnimationFrame(loop);
  }

  _render() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Draw dark space background grid
    this._renderGrid(ctx);

    // Draw document cluster points
    this._renderPoints(ctx);

    // Draw query point & proximity filaments
    if (this.queryPoint) {
      this._renderQueryAndFilaments(ctx);
    }
  }

  _renderGrid(ctx) {
    const cx = this.width / 2 + this.panX;
    const cy = this.height / 2 + this.panY;
    const gridSize = 40 * this.scale;

    ctx.save();
    ctx.strokeStyle = "rgba(32, 41, 56, 0.45)";
    ctx.lineWidth = 1;

    // Subtle coordinate rings
    [1, 2, 3].forEach((r) => {
      const radius = r * 75 * this.scale;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, this.height);
    ctx.moveTo(0, cy);
    ctx.lineTo(this.width, cy);
    ctx.stroke();
    ctx.restore();
  }

  _renderPoints(ctx) {
    for (const p of this.points) {
      if (this.activeCluster !== "all" && p.domain !== this.activeCluster) continue;

      const { sx, sy } = this._worldToScreen(p.x, p.y);
      const isRetrieved = this.retrievedIds.has(p.id) || this.retrievedIds.has(String(p.faiss_id));
      const isHovered = this.hoveredPoint === p;
      const color = this.domainColors[p.domain] || this.domainColors.default;

      ctx.save();

      if (isRetrieved) {
        // High-contrast golden/cyan glow for retrieved nodes
        const pulse = 1 + 0.18 * Math.sin(this.animTime * 3);
        ctx.shadowColor = "#22D3EE";
        ctx.shadowBlur = 16 * pulse;
        ctx.fillStyle = "#22D3EE";
        ctx.beginPath();
        ctx.arc(sx, sy, 7 * pulse, 0, Math.PI * 2);
        ctx.fill();

        // Inner white core
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, Math.PI * 2);
        ctx.fill();

        // Target ring
        ctx.strokeStyle = "rgba(34, 211, 238, 0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 12 * pulse, 0, Math.PI * 2);
        ctx.stroke();
      } else if (isHovered) {
        // Hovered node
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Standard corpus node
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.78;
        ctx.beginPath();
        ctx.arc(sx, sy, 3.8 * Math.min(1.4, Math.max(0.7, this.scale)), 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
    }
  }

  _renderQueryAndFilaments(ctx) {
    const qScreen = this._worldToScreen(this.queryPoint.x, this.queryPoint.y);

    // 1. Draw animated connecting filaments to retrieved nodes
    for (const p of this.points) {
      if (this.retrievedIds.has(p.id) || this.retrievedIds.has(String(p.faiss_id))) {
        const pScreen = this._worldToScreen(p.x, p.y);

        ctx.save();
        const grad = ctx.createLinearGradient(qScreen.sx, qScreen.sy, pScreen.sx, pScreen.sy);
        grad.addColorStop(0, "rgba(199, 165, 255, 0.95)");
        grad.addColorStop(0.5, "rgba(121, 162, 246, 0.75)");
        grad.addColorStop(1, "rgba(0, 155, 220, 0.45)");

        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 4]);
        ctx.lineDashOffset = -this.animTime * 18;

        ctx.beginPath();
        ctx.moveTo(qScreen.sx, qScreen.sy);
        ctx.lineTo(pScreen.sx, pScreen.sy);
        ctx.stroke();
        ctx.restore();
      }
    }

    // 2. Draw animated radar ripples from Query
    ctx.save();
    for (let i = 0; i < 3; i++) {
      const ringPhase = (this.animTime * 0.8 + i * 0.33) % 1;
      const radius = ringPhase * 60;
      const alpha = (1 - ringPhase) * 0.65;

      ctx.strokeStyle = `rgba(121, 162, 246, ${alpha})`;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(qScreen.sx, qScreen.sy, radius, 0, Math.PI * 2);
      ctx.stroke();
    }

    // 3. Draw Query Star / Anchor Node
    ctx.shadowColor = "#c7a5ff";
    ctx.shadowBlur = 24;
    ctx.fillStyle = "#79a2f6";
    ctx.beginPath();
    ctx.arc(qScreen.sx, qScreen.sy, 9, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(qScreen.sx, qScreen.sy, 4.5, 0, Math.PI * 2);
    ctx.fill();

    // Query text badge
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "600 11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("QUERY", qScreen.sx, qScreen.sy - 15);
    ctx.restore();
  }
}

window.VectorSpaceVisualizer = VectorSpaceVisualizer;
