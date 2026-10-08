/**
 * VectorIQ — High-Precision Performance Gauges Library
 * Implements pure DOM + CSS + SVG instruments:
 * 1. Tachometer (0–7000 r/min with redline band, machined oil-temp sub-dial, counterweighted needle)
 * 2. Speedometer (0–160 km/h / ms scale, travelled arc fill, peak-hold marker, digital readout, rolling odometer)
 * 3. Turbo Boost (-1.0 to 1.5 bar, vacuum/pressure separation, pulsing overboost lamp)
 * 4. EV Power (regeneration-to-power scale from zero, green charge segment, kilowatt readout, battery SOC bar)
 */

class PerformanceGauge {
  constructor(containerId, options = {}) {
    this.container = typeof containerId === "string" ? document.getElementById(containerId) : containerId;
    if (!this.container) return;

    this.variant = options.variant || "tachometer";
    this.min = options.min !== undefined ? options.min : 0;
    this.max = options.max !== undefined ? options.max : 7000;
    this.value = options.value !== undefined ? options.value : 0;
    this.peakValue = options.peakValue !== undefined ? options.peakValue : this.value;
    this.unit = options.unit || (this.variant === "tachometer" ? "r/min" : this.variant === "speedometer" ? "ms" : this.variant === "boost" ? "bar" : "kW");
    this.title = options.title || (this.variant === "tachometer" ? "Query Velocity" : this.variant === "speedometer" ? "Retrieval Latency" : this.variant === "boost" ? "Vector Pressure" : "Compute Power");
    this.subtitle = options.subtitle || "";

    // Polar geometry angles (degrees): 240 deg total sweep
    this.startAngle = options.startAngle !== undefined ? options.startAngle : -210;
    this.endAngle = options.endAngle !== undefined ? options.endAngle : 30;
    this.totalAngle = this.endAngle - this.startAngle;

    this.render();
  }

  // Map value to angle
  _valToAngle(val) {
    const clamped = Math.max(this.min, Math.min(this.max, val));
    const ratio = (clamped - this.min) / (this.max - this.min);
    return this.startAngle + ratio * this.totalAngle;
  }

  // Polar coordinates on 200x200 SVG viewBox
  _polarToCartesian(cx, cy, r, angleDeg) {
    const rad = ((angleDeg - 90) * Math.PI) / 180.0;
    return {
      x: cx + r * Math.cos(rad),
      y: cy + r * Math.sin(rad),
    };
  }

  _describeArc(cx, cy, r, startAngle, endAngle) {
    const start = this._polarToCartesian(cx, cy, r, endAngle);
    const end = this._polarToCartesian(cx, cy, r, startAngle);
    const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
    return ["M", start.x, start.y, "A", r, r, 0, largeArcFlag, 0, end.x, end.y].join(" ");
  }

  render() {
    this.container.innerHTML = "";
    const card = document.createElement("div");
    card.className = "gauge-instrument-card";

    // Header
    card.innerHTML = `
      <div class="gauge-instrument-header">
        <span class="gauge-instrument-title">
          <span>${this._getIcon()}</span>
          <span>${this.title}</span>
        </span>
        <span class="gauge-instrument-badge" id="badge_${this.variant}">${this.subtitle || "REAL-TIME"}</span>
      </div>
    `;

    // Outer Bezel & Dial
    const bezel = document.createElement("div");
    bezel.className = "gauge-bezel";

    const dial = document.createElement("div");
    dial.className = `gauge-dial variant-${this.variant}`;

    // Lens reflection
    const lens = document.createElement("div");
    lens.className = "gauge-lens";
    dial.appendChild(lens);

    // SVG Dial Graphics
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 200 200");
    svg.setAttribute("class", "gauge-svg");

    this._renderSvgContent(svg);
    dial.appendChild(svg);

    // Sub-dial (for Tachometer / Boost / EV)
    if (this.variant === "tachometer") {
      const subdial = document.createElement("div");
      subdial.className = "gauge-subdial";
      subdial.innerHTML = `
        <span class="gauge-subdial-label">INDEX</span>
        <span class="gauge-subdial-val font-mono">v002</span>
      `;
      dial.appendChild(subdial);
    } else if (this.variant === "boost") {
      const subdial = document.createElement("div");
      subdial.className = "gauge-subdial";
      subdial.innerHTML = `
        <span class="gauge-subdial-label" style="display:flex; align-items:center; gap:3px;">
          <span class="gauge-lamp" id="lamp_${this.variant}"></span>
          <span>BOOST</span>
        </span>
        <span class="gauge-subdial-val font-mono" id="boostStatusText">NOMINAL</span>
      `;
      dial.appendChild(subdial);
    }

    // Chrome Center Pivot
    const pivot = document.createElement("div");
    pivot.className = "gauge-pivot";
    dial.appendChild(pivot);

    // Needle Element (CSS rotated)
    const needleGroup = document.createElement("div");
    needleGroup.className = "gauge-needle-group";
    needleGroup.id = `needle_${this.variant}`;
    const initialAngle = this._valToAngle(this.value);
    needleGroup.style.transform = `rotate(${initialAngle}deg)`;

    needleGroup.innerHTML = `
      <svg viewBox="0 0 200 200" class="gauge-needle-svg">
        <defs>
          <linearGradient id="needleGrad_${this.variant}" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stop-color="#ef4444" />
            <stop offset="50%" stop-color="#f87171" />
            <stop offset="100%" stop-color="#dc2626" />
          </linearGradient>
          <filter id="needleShadow_${this.variant}" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="2" dy="4" stdDeviation="3" flood-color="rgba(0,0,0,0.7)" />
          </filter>
        </defs>
        <!-- Tapered Needle Blade with Counterweight Hub -->
        <g filter="url(#needleShadow_${this.variant})">
          <polygon points="98,100 100,22 102,100 101,126 99,126" fill="url(#needleGrad_${this.variant})" />
          <circle cx="100" cy="120" r="5" fill="#1e293b" />
          <circle cx="100" cy="100" r="8" fill="#e2e8f0" />
        </g>
      </svg>
    `;
    dial.appendChild(needleGroup);

    bezel.appendChild(dial);
    card.appendChild(bezel);

    // Digital Readout
    const readout = document.createElement("div");
    readout.className = "gauge-digital-readout";
    readout.innerHTML = `
      <div class="gauge-digital-val">
        <span id="readout_val_${this.variant}">${this._formatVal(this.value)}</span>
        <span class="gauge-digital-unit">${this.unit}</span>
      </div>
      <div class="gauge-digital-caption" id="readout_sub_${this.variant}">
        ${this._getReadoutCaption()}
      </div>
    `;

    // Mechanical Odometer (for Speedometer)
    if (this.variant === "speedometer") {
      const odo = document.createElement("div");
      odo.className = "gauge-odometer";
      odo.id = `odo_${this.variant}`;
      odo.innerHTML = `
        <span class="gauge-odometer-digit">0</span>
        <span class="gauge-odometer-digit">0</span>
        <span class="gauge-odometer-digit">1</span>
        <span class="gauge-odometer-digit">0</span>
        <span class="gauge-odometer-digit">0</span>
        <span class="gauge-odometer-digit">0</span>
      `;
      readout.appendChild(odo);
    }

    // Battery SOC bar (for EV Power)
    if (this.variant === "power") {
      const soc = document.createElement("div");
      soc.className = "gauge-soc-bar";
      soc.innerHTML = `<div class="gauge-soc-fill" id="soc_fill_${this.variant}" style="width: 78%;"></div>`;
      readout.appendChild(soc);
    }

    card.appendChild(readout);
    this.container.appendChild(card);

    this.needleEl = needleGroup;
    this.readoutEl = document.getElementById(`readout_val_${this.variant}`);
    this.trackEl = document.getElementById(`track_${this.variant}`);
    this.peakMarkerEl = document.getElementById(`peak_${this.variant}`);
    this.lampEl = document.getElementById(`lamp_${this.variant}`);
    this.socFillEl = document.getElementById(`soc_fill_${this.variant}`);

    // Initial update
    this.setValue(this.value, { updatePeak: true });
  }

  _getIcon() {
    switch (this.variant) {
      case "tachometer": return "⚡";
      case "speedometer": return "⏱";
      case "boost": return "▲";
      case "power": return "🔋";
      default: return "●";
    }
  }

  _getReadoutCaption() {
    switch (this.variant) {
      case "tachometer": return "Engine Throughput";
      case "speedometer": return "P95 Calibrated Latency";
      case "boost": return "FAISS Surge Capacity";
      case "power": return "Cache & Vector Compute";
      default: return "";
    }
  }

  _formatVal(v) {
    if (this.variant === "speedometer") return v.toFixed(1);
    if (this.variant === "boost") return v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2);
    if (this.variant === "tachometer") return Math.round(v).toLocaleString();
    return Math.round(v).toString();
  }

  _renderSvgContent(svg) {
    const cx = 100;
    const cy = 100;
    const r = 78;

    // Background track arc
    const bgArc = document.createElementNS("http://www.w3.org/2000/svg", "path");
    bgArc.setAttribute("d", this._describeArc(cx, cy, r, this.startAngle, this.endAngle));
    bgArc.setAttribute("class", "gauge-track-bg");
    svg.appendChild(bgArc);

    // Active Travelled Track Arc
    const travelledArc = document.createElementNS("http://www.w3.org/2000/svg", "path");
    travelledArc.setAttribute("id", `track_${this.variant}`);
    travelledArc.setAttribute("d", this._describeArc(cx, cy, r, this.startAngle, this.endAngle));
    travelledArc.setAttribute("class", "gauge-track-travelled");

    // Gradients
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `
      <linearGradient id="trackGrad_${this.variant}" x1="0%" y1="100%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#009bdc" />
        <stop offset="50%" stop-color="#79a2f6" />
        <stop offset="100%" stop-color="#c7a5ff" />
      </linearGradient>
    `;
    svg.appendChild(defs);

    travelledArc.setAttribute("stroke", `url(#trackGrad_${this.variant})`);

    // Setup stroke dasharray for arc filling
    const arcLen = (Math.PI * r * this.totalAngle) / 180;
    travelledArc.style.strokeDasharray = `${arcLen} ${arcLen}`;
    travelledArc.style.strokeDashoffset = `${arcLen}`;
    svg.appendChild(travelledArc);
    this.arcLength = arcLen;

    // Render Ticks & Labels
    this._renderTicks(svg, cx, cy, r);
  }

  _renderTicks(svg, cx, cy, r) {
    let tickCount = 10;
    let labelStep = 1;
    let labelMultiplier = 1;
    let redlineStart = Infinity;

    if (this.variant === "tachometer") {
      tickCount = 35; // 0 to 7000 (steps of 200)
      labelStep = 5; // major every 1000
      labelMultiplier = 1000;
      redlineStart = 5500;
    } else if (this.variant === "speedometer") {
      tickCount = 32; // 0 to 160 (steps of 5)
      labelStep = 4; // major every 20
      labelMultiplier = 1;
    } else if (this.variant === "boost") {
      tickCount = 25; // -1.0 to 1.5 (steps of 0.1)
      labelStep = 5; // major every 0.5
      labelMultiplier = 1;
      redlineStart = 1.0;
    } else if (this.variant === "power") {
      tickCount = 20; // -100 to 100
      labelStep = 5;
      labelMultiplier = 1;
    }

    for (let i = 0; i <= tickCount; i++) {
      const ratio = i / tickCount;
      const angle = this.startAngle + ratio * this.totalAngle;
      const val = this.min + ratio * (this.max - this.min);
      const isMajor = i % labelStep === 0;
      const isRedline = val >= redlineStart;

      const tickLen = isMajor ? 8 : 4;
      const p1 = this._polarToCartesian(cx, cy, r - 6, angle);
      const p2 = this._polarToCartesian(cx, cy, r - 6 - tickLen, angle);

      const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
      tick.setAttribute("x1", p1.x);
      tick.setAttribute("y1", p1.y);
      tick.setAttribute("x2", p2.x);
      tick.setAttribute("y2", p2.y);
      tick.setAttribute("class", isRedline ? "gauge-tick-redline" : isMajor ? "gauge-tick-major" : "gauge-tick");
      svg.appendChild(tick);

      if (isMajor) {
        const lp = this._polarToCartesian(cx, cy, r - 22, angle);
        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", lp.x);
        text.setAttribute("y", lp.y);
        text.setAttribute("class", isRedline ? "gauge-tick-label redline" : "gauge-tick-label");

        let displayVal = val;
        if (this.variant === "tachometer") displayVal = Math.round(val / 1000);
        else if (this.variant === "boost") displayVal = val.toFixed(1);
        else displayVal = Math.round(val);

        text.textContent = String(displayVal);
        svg.appendChild(text);
      }
    }

    // Dial Center Labels
    const centerText = document.createElementNS("http://www.w3.org/2000/svg", "text");
    centerText.setAttribute("x", cx);
    centerText.setAttribute("y", cy - 25);
    centerText.setAttribute("class", "gauge-tick-label");
    centerText.setAttribute("style", "font-size: 8px; fill: #64748b; font-weight: 700; letter-spacing: 0.08em;");
    centerText.textContent = this.variant === "tachometer" ? "RPM x1000" : this.variant === "speedometer" ? "KM/H • MS" : this.variant === "boost" ? "PRESSURE BAR" : "POWER %";
    svg.appendChild(centerText);

    // Peak-hold marker triangle
    const peakMarker = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    peakMarker.setAttribute("id", `peak_${this.variant}`);
    peakMarker.setAttribute("points", "-3,0 3,0 0,6");
    peakMarker.setAttribute("class", "gauge-peak-marker");
    svg.appendChild(peakMarker);
  }

  // Update needle, arc, and readouts
  setValue(val, options = {}) {
    this.value = Math.max(this.min, Math.min(this.max, val));
    if (options.updatePeak !== false && this.value > this.peakValue) {
      this.peakValue = this.value;
    }

    const angle = this._valToAngle(this.value);

    // 1. Needle Rotation
    if (this.needleEl) {
      this.needleEl.style.transform = `rotate(${angle}deg)`;
    }

    // 2. Travelled Arc Fill
    if (this.trackEl && this.arcLength) {
      const ratio = (this.value - this.min) / (this.max - this.min);
      const offset = this.arcLength * (1 - ratio);
      this.trackEl.style.strokeDashoffset = `${offset}`;
    }

    // 3. Peak-Hold Marker Position
    if (this.peakMarkerEl) {
      const peakAngle = this._valToAngle(this.peakValue);
      const p = this._polarToCartesian(100, 100, 78 + 3, peakAngle);
      this.peakMarkerEl.setAttribute("transform", `translate(${p.x}, ${p.y}) rotate(${peakAngle + 90})`);
    }

    // 4. Digital Readout
    if (this.readoutEl) {
      this.readoutEl.textContent = this._formatVal(this.value);
    }

    // 5. Overboost lamp
    if (this.lampEl) {
      this.lampEl.classList.toggle("active", this.value >= 1.0);
      const statusText = document.getElementById("boostStatusText");
      if (statusText) {
        statusText.innerText = this.value >= 1.0 ? "OVERBOOST" : "NOMINAL";
        statusText.style.color = this.value >= 1.0 ? "#ef4444" : "var(--brand-blue)";
      }
    }

    // 6. SOC Bar
    if (this.socFillEl && this.variant === "power") {
      const socRatio = Math.max(10, Math.min(100, Math.round(((this.value - this.min) / (this.max - this.min)) * 100)));
      this.socFillEl.style.width = `${socRatio}%`;
    }
  }

  // Simulate needle spring throttle burst / rev test
  revTest(targetVal, duration = 1200) {
    const origVal = this.value;
    const peak = targetVal !== undefined ? targetVal : this.max * 0.88;

    // Spring up
    this.setValue(peak, { updatePeak: true });

    setTimeout(() => {
      // Settle back to original
      this.setValue(origVal, { updatePeak: false });
    }, duration);
  }
}

// Global Performance Gauges Registry & Coordinator
const PerformanceGaugesCluster = {
  gauges: {},

  init(containerSelector = "#gaugeClusterContainer") {
    const container = document.querySelector(containerSelector);
    if (!container) return;

    container.innerHTML = `
      <div class="gauge-cluster">
        <div id="gauge_tachometer"></div>
        <div id="gauge_speedometer"></div>
        <div id="gauge_boost"></div>
        <div id="gauge_power"></div>
      </div>
    `;

    // 1. Tachometer: 0–7000 QPM/RPM, redline band
    this.gauges.tachometer = new PerformanceGauge("gauge_tachometer", {
      variant: "tachometer",
      min: 0,
      max: 7000,
      value: 1200,
      unit: "QPM",
      title: "Query Velocity",
      subtitle: "BURST ENGINE",
    });

    // 2. Speedometer: 0–160 ms Latency, travelled arc fill, peak hold, odometer
    this.gauges.speedometer = new PerformanceGauge("gauge_speedometer", {
      variant: "speedometer",
      min: 0,
      max: 160,
      value: 5.9,
      peakValue: 8.4,
      unit: "ms",
      title: "Retrieval Latency",
      subtitle: "P95 CALIBRATED",
    });

    // 3. Turbo Boost: -1.0 to 1.5 bar Vector Queue Pressure
    this.gauges.boost = new PerformanceGauge("gauge_boost", {
      variant: "boost",
      min: -1.0,
      max: 1.5,
      value: 0.15,
      unit: "bar",
      title: "Vector Pressure",
      subtitle: "QUEUE SURGE",
    });

    // 4. EV Power: -100 to 100% Cache Efficiency & Dense Compute
    this.gauges.power = new PerformanceGauge("gauge_power", {
      variant: "power",
      min: -100,
      max: 100,
      value: 68,
      unit: "%",
      title: "Compute Power",
      subtitle: "CACHE & 512D RAM",
    });
  },

  // Fire animated sweep across all 4 gauges
  runRevTest() {
    if (this.gauges.tachometer) this.gauges.tachometer.revTest(6400, 1400);
    if (this.gauges.speedometer) this.gauges.speedometer.revTest(72.5, 1400);
    if (this.gauges.boost) this.gauges.boost.revTest(1.25, 1400);
    if (this.gauges.power) this.gauges.power.revTest(95, 1400);
  },

  // Update speedometer to exact query latency
  onQueryComplete(latencyMs) {
    if (this.gauges.speedometer) {
      this.gauges.speedometer.setValue(latencyMs, { updatePeak: true });
    }
    if (this.gauges.tachometer) {
      // Simulate instantaneous query burst
      const burstQpm = Math.min(6800, 2400 + Math.random() * 3000);
      this.gauges.tachometer.revTest(burstQpm, 900);
    }
    if (this.gauges.boost) {
      const pressure = Math.min(1.4, (latencyMs / 50.0) * 0.8);
      this.gauges.boost.setValue(pressure, { updatePeak: true });
    }
  },
};
