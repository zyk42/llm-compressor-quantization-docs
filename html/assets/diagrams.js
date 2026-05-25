// SVG Diagrams for LLM Quantization Documentation
// Theme: bg #0f172a, text #f1f5f9, primary #6366f1, success #10b981, accent #f59e0b

const DIAGRAMS = {

  quantization_overview: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200" width="600" height="200">
  <defs>
    <marker id="qo-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="600" height="200" fill="#0f172a" rx="8"/>
  <!-- FP16 Model -->
  <rect x="20" y="60" width="110" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="75" y="82" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="12" font-weight="bold">FP16 Model</text>
  <text x="75" y="100" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="10">16 bits/param</text>
  <!-- Arrow 1 -->
  <line x1="130" y1="85" x2="165" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#qo-arrow)"/>
  <!-- Quantization -->
  <rect x="170" y="55" width="120" height="60" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="230" y="80" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="12" font-weight="bold">Quantization</text>
  <text x="230" y="98" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="10">GPTQ/AWQ/FP8</text>
  <!-- Arrow 2 -->
  <line x1="290" y1="85" x2="325" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#qo-arrow)"/>
  <!-- Compressed Model -->
  <rect x="330" y="55" width="120" height="60" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="390" y="77" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="12" font-weight="bold">INT4/FP8</text>
  <text x="390" y="93" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="11">Model</text>
  <text x="390" y="108" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="10">4-8 bits/param</text>
  <!-- Arrow 3 -->
  <line x1="450" y1="85" x2="485" y2="85" stroke="#6366f1" stroke-width="2" marker-end="url(#qo-arrow)"/>
  <!-- vLLM Inference -->
  <rect x="490" y="60" width="90" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="535" y="82" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="12" font-weight="bold">vLLM</text>
  <text x="535" y="98" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="10">Inference</text>
  <!-- Metrics -->
  <rect x="150" y="140" width="130" height="35" rx="6" fill="#10b98120" stroke="#10b981" stroke-width="1"/>
  <text x="215" y="162" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">2-4x Compression</text>
  <rect x="320" y="140" width="140" height="35" rx="6" fill="#6366f120" stroke="#6366f1" stroke-width="1"/>
  <text x="390" y="162" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="11" font-weight="bold">1.5-3x Speedup</text>
</svg>`,

  architecture_pipeline: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="600" height="220">
  <defs>
    <marker id="ap-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="600" height="220" fill="#0f172a" rx="8"/>
  <!-- Title -->
  <text x="300" y="25" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">LLM Compressor Pipeline Architecture</text>
  <!-- Recipe -->
  <rect x="20" y="50" width="100" height="55" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="70" y="73" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="11" font-weight="bold">Recipe</text>
  <text x="70" y="90" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">YAML config</text>
  <text x="70" y="102" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">quantization spec</text>
  <!-- Arrow -->
  <line x1="120" y1="77" x2="148" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#ap-arrow)"/>
  <!-- Modifier -->
  <rect x="153" y="50" width="100" height="55" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="203" y="73" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="11" font-weight="bold">Modifier</text>
  <text x="203" y="90" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">GPTQModifier</text>
  <text x="203" y="102" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">SmoothQuantModifier</text>
  <!-- Arrow -->
  <line x1="253" y1="77" x2="281" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#ap-arrow)"/>
  <!-- Pipeline -->
  <rect x="286" y="50" width="100" height="55" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="336" y="73" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="11" font-weight="bold">Pipeline</text>
  <text x="336" y="90" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Sequential layer</text>
  <text x="336" y="102" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">processing</text>
  <!-- Arrow -->
  <line x1="386" y1="77" x2="414" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#ap-arrow)"/>
  <!-- Observer -->
  <rect x="419" y="50" width="100" height="55" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="469" y="73" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">Observer</text>
  <text x="469" y="90" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Collect stats</text>
  <text x="469" y="102" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">min/max/scale</text>
  <!-- Arrow down -->
  <line x1="469" y1="105" x2="469" y2="133" stroke="#6366f1" stroke-width="2" marker-end="url(#ap-arrow)"/>
  <!-- Output -->
  <rect x="380" y="138" width="180" height="55" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="470" y="161" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">compressed-tensors</text>
  <text x="470" y="178" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">SafeTensors + config.json</text>
  <text x="470" y="190" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">HuggingFace compatible</text>
  <!-- Calibration data arrow -->
  <rect x="20" y="138" width="120" height="50" rx="8" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="80" y="161" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="10">Calibration Data</text>
  <text x="80" y="177" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">512 samples</text>
  <line x1="140" y1="163" x2="280" y2="100" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4" marker-end="url(#ap-arrow)"/>
</svg>`,

  rtn_process: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 180" width="600" height="180">
  <defs>
    <marker id="rtn-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="600" height="180" fill="#0f172a" rx="8"/>
  <text x="300" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">RTN (Round-To-Nearest) Quantization</text>
  <!-- Step 1: Weight -->
  <rect x="15" y="45" width="85" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="57" y="67" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="11" font-weight="bold">Weight W</text>
  <text x="57" y="83" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">FP16 values</text>
  <!-- Arrow -->
  <line x1="100" y1="70" x2="123" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#rtn-arrow)"/>
  <!-- Step 2: Compute Scale -->
  <rect x="128" y="45" width="95" height="50" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="175" y="64" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Compute Scale</text>
  <text x="175" y="80" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">s = max|W|</text>
  <text x="175" y="92" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">/ (2^(b-1)-1)</text>
  <!-- Arrow -->
  <line x1="223" y1="70" x2="246" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#rtn-arrow)"/>
  <!-- Step 3: Round -->
  <rect x="251" y="45" width="85" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="293" y="65" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="10" font-weight="bold">Round</text>
  <text x="293" y="82" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">round(W / s)</text>
  <!-- Arrow -->
  <line x1="336" y1="70" x2="359" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#rtn-arrow)"/>
  <!-- Step 4: Clamp -->
  <rect x="364" y="45" width="95" height="50" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="411" y="65" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Clamp</text>
  <text x="411" y="82" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">clip(-2^(b-1),</text>
  <text x="411" y="92" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">2^(b-1)-1)</text>
  <!-- Arrow -->
  <line x1="459" y1="70" x2="482" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#rtn-arrow)"/>
  <!-- Step 5: Quantized -->
  <rect x="487" y="45" width="98" height="50" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="536" y="65" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="10" font-weight="bold">Quantized W</text>
  <text x="536" y="82" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">INT4/INT8</text>
  <!-- Formula summary -->
  <rect x="100" y="120" width="400" height="40" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="300" y="138" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="11">W_q = clamp(round(W / scale), -Q_n, Q_p)</text>
  <text x="300" y="153" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Dequantize: W_hat = W_q * scale (introduces quantization error)</text>
</svg>`,

  gptq_flow: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 250" width="600" height="250">
  <defs>
    <marker id="gptq-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
    <marker id="gptq-arrow-red" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#ef4444"/>
    </marker>
  </defs>
  <rect width="600" height="250" fill="#0f172a" rx="8"/>
  <text x="300" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">GPTQ Algorithm Flow</text>
  <!-- Calibration Data -->
  <rect x="20" y="45" width="110" height="50" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="75" y="66" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Calibration Data</text>
  <text x="75" y="82" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">X (128-512 samples)</text>
  <!-- Arrow -->
  <line x1="130" y1="70" x2="158" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#gptq-arrow)"/>
  <!-- Hessian -->
  <rect x="163" y="45" width="120" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="223" y="64" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Hessian Matrix</text>
  <text x="223" y="80" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="10">H = 2X^T X</text>
  <text x="223" y="92" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">weight sensitivity</text>
  <!-- Arrow -->
  <line x1="283" y1="70" x2="311" y2="70" stroke="#6366f1" stroke-width="2" marker-end="url(#gptq-arrow)"/>
  <!-- Cholesky -->
  <rect x="316" y="45" width="120" height="50" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="376" y="64" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Cholesky Inverse</text>
  <text x="376" y="80" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="10">H^(-1) = LL^T</text>
  <text x="376" y="92" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">error distribution</text>
  <!-- Arrow down -->
  <line x1="376" y1="95" x2="376" y2="123" stroke="#6366f1" stroke-width="2" marker-end="url(#gptq-arrow)"/>
  <!-- Column-by-column -->
  <rect x="80" y="128" width="450" height="70" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="305" y="148" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">Column-by-Column Quantization (block size=128)</text>
  <!-- Column blocks -->
  <rect x="100" y="158" width="50" height="28" rx="4" fill="#6366f130" stroke="#6366f1" stroke-width="1.5"/>
  <text x="125" y="176" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9">col 0</text>
  <rect x="160" y="158" width="50" height="28" rx="4" fill="#6366f130" stroke="#6366f1" stroke-width="1.5"/>
  <text x="185" y="176" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9">col 1</text>
  <rect x="220" y="158" width="50" height="28" rx="4" fill="#6366f130" stroke="#6366f1" stroke-width="1.5"/>
  <text x="245" y="176" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9">col 2</text>
  <text x="295" y="176" fill="#94a3b8" font-family="system-ui" font-size="12">...</text>
  <rect x="320" y="158" width="50" height="28" rx="4" fill="#10b98130" stroke="#10b981" stroke-width="1.5"/>
  <text x="345" y="176" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="9">col N</text>
  <!-- Error propagation arrows -->
  <path d="M 150 165 C 155 150, 160 150, 165 158" fill="none" stroke="#ef4444" stroke-width="1.5" marker-end="url(#gptq-arrow-red)"/>
  <path d="M 210 165 C 215 150, 220 150, 225 158" fill="none" stroke="#ef4444" stroke-width="1.5" marker-end="url(#gptq-arrow-red)"/>
  <path d="M 270 165 C 280 150, 310 150, 325 158" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3" marker-end="url(#gptq-arrow-red)"/>
  <!-- Error label -->
  <text x="230" y="147" text-anchor="middle" fill="#ef4444" font-family="system-ui" font-size="8">error propagation to remaining cols</text>
  <!-- Output -->
  <rect x="200" y="215" width="200" height="28" rx="6" fill="#10b98120" stroke="#10b981" stroke-width="1.5"/>
  <text x="300" y="234" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="10" font-weight="bold">Optimal INT4 Weights + Scales</text>
  <line x1="305" y1="198" x2="305" y2="213" stroke="#6366f1" stroke-width="2" marker-end="url(#gptq-arrow)"/>
</svg>`,

  awq_flow: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 200" width="600" height="200">
  <defs>
    <marker id="awq-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="600" height="200" fill="#0f172a" rx="8"/>
  <text x="300" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">AWQ (Activation-Aware Weight Quantization)</text>
  <!-- Step 1: Activations -->
  <rect x="15" y="50" width="95" height="55" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="62" y="70" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Activations</text>
  <text x="62" y="85" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Calibration</text>
  <text x="62" y="97" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">X samples</text>
  <!-- Arrow -->
  <line x1="110" y1="77" x2="133" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#awq-arrow)"/>
  <!-- Step 2: Salient Channels -->
  <rect x="138" y="45" width="105" height="65" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="190" y="63" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Find Salient</text>
  <text x="190" y="77" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Channels</text>
  <text x="190" y="93" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">top 1% by</text>
  <text x="190" y="104" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">magnitude</text>
  <!-- Arrow -->
  <line x1="243" y1="77" x2="266" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#awq-arrow)"/>
  <!-- Step 3: Grid Search -->
  <rect x="271" y="45" width="100" height="65" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="321" y="65" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Grid Search</text>
  <text x="321" y="82" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="10">alpha (0,1)</text>
  <text x="321" y="97" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">minimize</text>
  <text x="321" y="107" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">||Q(W*s)*X/s - WX||</text>
  <!-- Arrow -->
  <line x1="371" y1="77" x2="394" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#awq-arrow)"/>
  <!-- Step 4: Apply Scaling -->
  <rect x="399" y="50" width="90" height="55" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="444" y="70" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Apply Scale</text>
  <text x="444" y="86" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="9">W = W * s</text>
  <text x="444" y="99" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">protect salient</text>
  <!-- Arrow -->
  <line x1="489" y1="77" x2="507" y2="77" stroke="#6366f1" stroke-width="2" marker-end="url(#awq-arrow)"/>
  <!-- Step 5: Quantize -->
  <rect x="512" y="50" width="75" height="55" rx="8" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="549" y="72" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="10" font-weight="bold">Quantize</text>
  <text x="549" y="88" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">INT4</text>
  <text x="549" y="100" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">per-group</text>
  <!-- Key insight -->
  <rect x="120" y="140" width="360" height="40" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="300" y="157" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="10">Key Insight: 1% of salient weight channels disproportionately affect quality.</text>
  <text x="300" y="172" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Scaling protects these channels without mixed-precision overhead.</text>
</svg>`,

  smoothquant_transform: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 250" width="600" height="250">
  <defs>
    <marker id="sq-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="600" height="250" fill="#0f172a" rx="8"/>
  <text x="300" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">SmoothQuant: Activation-Weight Balancing</text>
  <!-- BEFORE section -->
  <text x="150" y="48" text-anchor="middle" fill="#ef4444" font-family="system-ui" font-size="11" font-weight="bold">BEFORE (Outlier Activations)</text>
  <!-- Spiky activation bars -->
  <rect x="40" y="110" width="12" height="20" rx="2" fill="#6366f180"/>
  <rect x="56" y="100" width="12" height="30" rx="2" fill="#6366f180"/>
  <rect x="72" y="55" width="12" height="75" rx="2" fill="#ef4444" stroke="#ef4444" stroke-width="1"/>
  <rect x="88" y="108" width="12" height="22" rx="2" fill="#6366f180"/>
  <rect x="104" y="105" width="12" height="25" rx="2" fill="#6366f180"/>
  <rect x="120" y="60" width="12" height="70" rx="2" fill="#ef4444" stroke="#ef4444" stroke-width="1"/>
  <rect x="136" y="112" width="12" height="18" rx="2" fill="#6366f180"/>
  <rect x="152" y="100" width="12" height="30" rx="2" fill="#6366f180"/>
  <rect x="168" y="107" width="12" height="23" rx="2" fill="#6366f180"/>
  <rect x="184" y="58" width="12" height="72" rx="2" fill="#ef4444" stroke="#ef4444" stroke-width="1"/>
  <rect x="200" y="106" width="12" height="24" rx="2" fill="#6366f180"/>
  <rect x="216" y="110" width="12" height="20" rx="2" fill="#6366f180"/>
  <text x="135" y="145" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Activation channels (outliers in red)</text>
  <!-- Baseline -->
  <line x1="35" y1="130" x2="235" y2="130" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="2"/>
  <!-- Transform arrow -->
  <rect x="260" y="75" width="80" height="60" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="300" y="95" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Smooth</text>
  <text x="300" y="110" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="9">X/s | W*s</text>
  <text x="300" y="125" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">s = |X|^a / |W|^(1-a)</text>
  <line x1="240" y1="105" x2="258" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#sq-arrow)"/>
  <line x1="340" y1="105" x2="358" y2="105" stroke="#6366f1" stroke-width="2" marker-end="url(#sq-arrow)"/>
  <!-- AFTER section -->
  <text x="460" y="48" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">AFTER (Smoothed)</text>
  <!-- Even activation bars -->
  <rect x="365" y="95" width="12" height="30" rx="2" fill="#10b98180"/>
  <rect x="381" y="92" width="12" height="33" rx="2" fill="#10b98180"/>
  <rect x="397" y="98" width="12" height="27" rx="2" fill="#10b98180"/>
  <rect x="413" y="90" width="12" height="35" rx="2" fill="#10b98180"/>
  <rect x="429" y="94" width="12" height="31" rx="2" fill="#10b98180"/>
  <rect x="445" y="96" width="12" height="29" rx="2" fill="#10b98180"/>
  <rect x="461" y="91" width="12" height="34" rx="2" fill="#10b98180"/>
  <rect x="477" y="97" width="12" height="28" rx="2" fill="#10b98180"/>
  <rect x="493" y="93" width="12" height="32" rx="2" fill="#10b98180"/>
  <rect x="509" y="95" width="12" height="30" rx="2" fill="#10b98180"/>
  <rect x="525" y="92" width="12" height="33" rx="2" fill="#10b98180"/>
  <rect x="541" y="96" width="12" height="29" rx="2" fill="#10b98180"/>
  <text x="460" y="145" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Smoothed activation channels (uniform)</text>
  <!-- Baseline -->
  <line x1="360" y1="125" x2="560" y2="125" stroke="#94a3b8" stroke-width="0.5" stroke-dasharray="2"/>
  <!-- Bottom explanation -->
  <rect x="60" y="170" width="200" height="55" rx="6" fill="#1e293b" stroke="#6366f1" stroke-width="1.5"/>
  <text x="160" y="190" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="10" font-weight="bold">Activations: X / s</text>
  <text x="160" y="207" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Reduce outlier magnitude</text>
  <text x="160" y="219" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Easier to quantize</text>
  <rect x="340" y="170" width="200" height="55" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="440" y="190" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="10" font-weight="bold">Weights: W * s</text>
  <text x="440" y="207" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Absorb difficulty into weights</text>
  <text x="440" y="219" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Weights are easier to quantize</text>
  <!-- Math equivalence -->
  <text x="300" y="245" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="10">Y = XW = (X diag(s)^(-1)) (diag(s) W) = X_smooth * W_smooth</text>
</svg>`,

  fp8_format: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 550 220" width="550" height="220">
  <rect width="550" height="220" fill="#0f172a" rx="8"/>
  <text x="275" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">FP8 vs INT8 Bit Layout</text>
  <!-- FP8 E4M3 -->
  <text x="30" y="55" fill="#f59e0b" font-family="system-ui" font-size="11" font-weight="bold">FP8 E4M3</text>
  <text x="200" y="55" fill="#94a3b8" font-family="system-ui" font-size="9">(range: +/-448, precision: 0.125)</text>
  <!-- Sign bit -->
  <rect x="30" y="65" width="45" height="35" rx="4" fill="#ef444440" stroke="#ef4444" stroke-width="2"/>
  <text x="52" y="86" text-anchor="middle" fill="#ef4444" font-family="monospace" font-size="12" font-weight="bold">S</text>
  <text x="52" y="112" text-anchor="middle" fill="#ef4444" font-family="system-ui" font-size="8">1 bit</text>
  <!-- Exponent bits -->
  <rect x="80" y="65" width="180" height="35" rx="4" fill="#6366f140" stroke="#6366f1" stroke-width="2"/>
  <text x="170" y="86" text-anchor="middle" fill="#6366f1" font-family="monospace" font-size="12" font-weight="bold">E E E E</text>
  <text x="170" y="112" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="8">4 exponent bits (bias=7)</text>
  <!-- Mantissa bits -->
  <rect x="265" y="65" width="135" height="35" rx="4" fill="#10b98140" stroke="#10b981" stroke-width="2"/>
  <text x="332" y="86" text-anchor="middle" fill="#10b981" font-family="monospace" font-size="12" font-weight="bold">M M M</text>
  <text x="332" y="112" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="8">3 mantissa bits</text>
  <!-- Formula -->
  <text x="440" y="86" fill="#94a3b8" font-family="monospace" font-size="9">= (-1)^S *</text>
  <text x="440" y="100" fill="#94a3b8" font-family="monospace" font-size="9">2^(E-7) * 1.M</text>
  <!-- INT8 -->
  <text x="30" y="148" fill="#f59e0b" font-family="system-ui" font-size="11" font-weight="bold">INT8</text>
  <text x="100" y="148" fill="#94a3b8" font-family="system-ui" font-size="9">(range: -128 to 127, uniform spacing)</text>
  <!-- Sign bit -->
  <rect x="30" y="158" width="45" height="35" rx="4" fill="#ef444440" stroke="#ef4444" stroke-width="2"/>
  <text x="52" y="179" text-anchor="middle" fill="#ef4444" font-family="monospace" font-size="12" font-weight="bold">S</text>
  <text x="52" y="205" text-anchor="middle" fill="#ef4444" font-family="system-ui" font-size="8">1 bit</text>
  <!-- Integer bits -->
  <rect x="80" y="158" width="320" height="35" rx="4" fill="#f59e0b40" stroke="#f59e0b" stroke-width="2"/>
  <text x="240" y="179" text-anchor="middle" fill="#f59e0b" font-family="monospace" font-size="12" font-weight="bold">I I I I I I I</text>
  <text x="240" y="205" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="8">7 integer bits (two's complement)</text>
  <!-- Comparison -->
  <rect x="430" y="140" width="105" height="60" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="482" y="158" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="9" font-weight="bold">Comparison</text>
  <text x="482" y="174" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="8">FP8: dynamic range</text>
  <text x="482" y="188" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="8">INT8: uniform steps</text>
</svg>`,

  kv_cache_flow: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="600" height="220">
  <defs>
    <marker id="kv-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
    <marker id="kv-arrow-green" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#10b981"/>
    </marker>
  </defs>
  <rect width="600" height="220" fill="#0f172a" rx="8"/>
  <text x="300" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">KV Cache Quantization Flow</text>
  <!-- Input -->
  <rect x="20" y="45" width="70" height="40" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="55" y="69" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="10" font-weight="bold">Input</text>
  <!-- Arrow -->
  <line x1="90" y1="65" x2="113" y2="65" stroke="#6366f1" stroke-width="2" marker-end="url(#kv-arrow)"/>
  <!-- Projections -->
  <rect x="118" y="38" width="85" height="55" rx="8" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="160" y="57" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9" font-weight="bold">Projections</text>
  <text x="160" y="72" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="9">q_proj</text>
  <text x="160" y="85" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="9">k_proj, v_proj</text>
  <!-- Arrows splitting -->
  <line x1="203" y1="55" x2="233" y2="45" stroke="#6366f1" stroke-width="1.5" marker-end="url(#kv-arrow)"/>
  <line x1="203" y1="75" x2="233" y2="85" stroke="#10b981" stroke-width="2" marker-end="url(#kv-arrow-green)"/>
  <!-- Q path (top) -->
  <rect x="238" y="30" width="55" height="30" rx="6" fill="#1e293b" stroke="#6366f1" stroke-width="1.5"/>
  <text x="265" y="49" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9" font-weight="bold">Q (FP16)</text>
  <!-- KV path -->
  <rect x="238" y="73" width="100" height="35" rx="6" fill="#10b98120" stroke="#10b981" stroke-width="2"/>
  <text x="288" y="88" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="9" font-weight="bold">K,V Quantize</text>
  <text x="288" y="101" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">FP16 -> FP8</text>
  <!-- Arrow to cache -->
  <line x1="338" y1="90" x2="363" y2="90" stroke="#10b981" stroke-width="2" marker-end="url(#kv-arrow-green)"/>
  <!-- KV Cache -->
  <rect x="368" y="60" width="90" height="65" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="413" y="80" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">KV Cache</text>
  <text x="413" y="95" text-anchor="middle" fill="#10b981" font-family="monospace" font-size="9">FP8 stored</text>
  <text x="413" y="110" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">50% memory</text>
  <text x="413" y="121" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">savings</text>
  <!-- Arrow from cache -->
  <line x1="458" y1="90" x2="483" y2="90" stroke="#6366f1" stroke-width="2" marker-end="url(#kv-arrow)"/>
  <!-- Dequantize -->
  <rect x="488" y="40" width="85" height="45" rx="6" fill="#1e293b" stroke="#6366f1" stroke-width="1.5"/>
  <text x="530" y="58" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9" font-weight="bold">Dequantize</text>
  <text x="530" y="73" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">FP8 -> FP16</text>
  <!-- Q connects to attention -->
  <line x1="293" y1="45" x2="500" y2="45" stroke="#6366f1" stroke-width="1" stroke-dasharray="4"/>
  <line x1="500" y1="45" x2="500" y2="95" stroke="#6366f1" stroke-width="1" stroke-dasharray="4"/>
  <!-- Attention -->
  <rect x="488" y="95" width="85" height="40" rx="6" fill="#1e293b" stroke="#6366f1" stroke-width="2"/>
  <text x="530" y="112" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="9" font-weight="bold">Attention</text>
  <text x="530" y="127" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">Q * K^T / sqrt(d)</text>
  <!-- Dequant to attention -->
  <line x1="530" y1="85" x2="530" y2="93" stroke="#6366f1" stroke-width="1.5" marker-end="url(#kv-arrow)"/>
  <!-- Output -->
  <rect x="488" y="148" width="85" height="30" rx="6" fill="#1e293b" stroke="#10b981" stroke-width="1.5"/>
  <text x="530" y="167" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="9" font-weight="bold">Output</text>
  <line x1="530" y1="135" x2="530" y2="146" stroke="#6366f1" stroke-width="1.5" marker-end="url(#kv-arrow)"/>
  <!-- Benefit box -->
  <rect x="20" y="155" width="280" height="48" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="160" y="173" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="9" font-weight="bold">Benefits: 2x longer sequences, 2x batch size</text>
  <text x="160" y="190" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">KV cache often dominates memory in long-context inference</text>
</svg>`,

  sequential_onloading: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 550 220" width="550" height="220">
  <defs>
    <marker id="so-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
    <marker id="so-arrow-acc" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#f59e0b"/>
    </marker>
  </defs>
  <rect width="550" height="220" fill="#0f172a" rx="8"/>
  <text x="275" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">Sequential Layer Onloading (Memory-Efficient Quantization)</text>
  <!-- GPU box -->
  <rect x="20" y="45" width="240" height="120" rx="10" fill="#1e293b" stroke="#10b981" stroke-width="2"/>
  <text x="140" y="65" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">GPU (VRAM)</text>
  <!-- Active layer -->
  <rect x="40" y="78" width="200" height="35" rx="6" fill="#6366f140" stroke="#6366f1" stroke-width="2"/>
  <text x="140" y="100" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="11" font-weight="bold">Layer 0 (Active - Quantizing)</text>
  <!-- Calibration running indicator -->
  <rect x="40" y="120" width="200" height="22" rx="4" fill="#f59e0b20" stroke="#f59e0b" stroke-width="1"/>
  <text x="140" y="135" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="9">Running calibration data through layer...</text>
  <!-- CPU/Disk box -->
  <rect x="300" y="45" width="230" height="120" rx="10" fill="#1e293b" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4"/>
  <text x="415" y="65" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="11" font-weight="bold">CPU / Disk (Waiting)</text>
  <!-- Waiting layers -->
  <rect x="315" y="78" width="200" height="22" rx="4" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="415" y="93" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Layer 1</text>
  <rect x="315" y="104" width="200" height="22" rx="4" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="415" y="119" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Layer 2</text>
  <rect x="315" y="130" width="200" height="22" rx="4" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="415" y="145" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Layer 3 ... Layer N</text>
  <!-- Transfer arrows -->
  <path d="M 260 100 C 280 100, 280 85, 300 85" fill="none" stroke="#f59e0b" stroke-width="2" marker-end="url(#so-arrow-acc)"/>
  <text x="283" y="78" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="8">offload</text>
  <path d="M 300 108 C 280 108, 280 120, 260 120" fill="none" stroke="#6366f1" stroke-width="2" marker-end="url(#so-arrow)"/>
  <text x="283" y="130" text-anchor="middle" fill="#6366f1" font-family="system-ui" font-size="8">load next</text>
  <!-- Timeline -->
  <rect x="20" y="180" width="510" height="30" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="35" y="199" fill="#94a3b8" font-family="system-ui" font-size="9">Time:</text>
  <rect x="75" y="188" width="70" height="14" rx="3" fill="#6366f1"/>
  <text x="110" y="199" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="8">Layer 0</text>
  <rect x="150" y="188" width="70" height="14" rx="3" fill="#6366f180"/>
  <text x="185" y="199" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="8">Layer 1</text>
  <rect x="225" y="188" width="70" height="14" rx="3" fill="#6366f160"/>
  <text x="260" y="199" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="8">Layer 2</text>
  <text x="310" y="199" fill="#94a3b8" font-family="system-ui" font-size="10">...</text>
  <rect x="330" y="188" width="70" height="14" rx="3" fill="#10b981"/>
  <text x="365" y="199" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="8">Layer N</text>
  <text x="425" y="199" fill="#10b981" font-family="system-ui" font-size="8" font-weight="bold">Done!</text>
</svg>`,

  rotation_transform: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 230" width="580" height="230">
  <defs>
    <marker id="rot-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#6366f1"/>
    </marker>
  </defs>
  <rect width="580" height="230" fill="#0f172a" rx="8"/>
  <text x="290" y="22" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="13" font-weight="bold">Rotation Transform (QuaRot / SpinQuant)</text>
  <!-- BEFORE label -->
  <text x="120" y="48" text-anchor="middle" fill="#ef4444" font-family="system-ui" font-size="11" font-weight="bold">Before: High Coherence (Outliers)</text>
  <!-- Before matrix - uneven coloring -->
  <g transform="translate(30, 55)">
    <rect x="0" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.2"/>
    <rect x="26" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.9"/>
    <rect x="52" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.15"/>
    <rect x="78" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="104" y="0" width="24" height="24" rx="2" fill="#ef4444" opacity="0.95"/>
    <rect x="130" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.2"/>
    <rect x="156" y="0" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="0" y="26" width="24" height="24" rx="2" fill="#6366f1" opacity="0.15"/>
    <rect x="26" y="26" width="24" height="24" rx="2" fill="#ef4444" opacity="0.9"/>
    <rect x="52" y="26" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="78" y="26" width="24" height="24" rx="2" fill="#6366f1" opacity="0.2"/>
    <rect x="104" y="26" width="24" height="24" rx="2" fill="#ef4444" opacity="0.85"/>
    <rect x="130" y="26" width="24" height="24" rx="2" fill="#6366f1" opacity="0.15"/>
    <rect x="156" y="26" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="0" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="26" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.85"/>
    <rect x="52" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="78" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.15"/>
    <rect x="104" y="52" width="24" height="24" rx="2" fill="#ef4444" opacity="0.92"/>
    <rect x="130" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.2"/>
    <rect x="156" y="52" width="24" height="24" rx="2" fill="#6366f1" opacity="0.12"/>
    <rect x="0" y="78" width="24" height="24" rx="2" fill="#6366f1" opacity="0.2"/>
    <rect x="26" y="78" width="24" height="24" rx="2" fill="#ef4444" opacity="0.88"/>
    <rect x="52" y="78" width="24" height="24" rx="2" fill="#6366f1" opacity="0.12"/>
    <rect x="78" y="78" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
    <rect x="104" y="78" width="24" height="24" rx="2" fill="#ef4444" opacity="0.9"/>
    <rect x="130" y="78" width="24" height="24" rx="2" fill="#6366f1" opacity="0.18"/>
    <rect x="156" y="78" width="24" height="24" rx="2" fill="#6366f1" opacity="0.1"/>
  </g>
  <text x="120" y="170" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Outlier columns dominate (hard to quantize)</text>
  <!-- Transform arrow -->
  <rect x="240" y="85" width="100" height="55" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="2"/>
  <text x="290" y="105" text-anchor="middle" fill="#f59e0b" font-family="system-ui" font-size="10" font-weight="bold">Rotation</text>
  <text x="290" y="120" text-anchor="middle" fill="#f1f5f9" font-family="monospace" font-size="9">X' = XR</text>
  <text x="290" y="133" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="8">R = Hadamard/Random</text>
  <line x1="215" y1="112" x2="238" y2="112" stroke="#6366f1" stroke-width="2" marker-end="url(#rot-arrow)"/>
  <line x1="340" y1="112" x2="363" y2="112" stroke="#6366f1" stroke-width="2" marker-end="url(#rot-arrow)"/>
  <!-- AFTER label -->
  <text x="460" y="48" text-anchor="middle" fill="#10b981" font-family="system-ui" font-size="11" font-weight="bold">After: Low Coherence (Uniform)</text>
  <!-- After matrix - even coloring -->
  <g transform="translate(370, 55)">
    <rect x="0" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.45"/>
    <rect x="26" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="52" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.42"/>
    <rect x="78" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.48"/>
    <rect x="104" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.44"/>
    <rect x="130" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="156" y="0" width="24" height="24" rx="2" fill="#10b981" opacity="0.46"/>
    <rect x="0" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.48"/>
    <rect x="26" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.43"/>
    <rect x="52" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="78" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.45"/>
    <rect x="104" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.47"/>
    <rect x="130" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.42"/>
    <rect x="156" y="26" width="24" height="24" rx="2" fill="#10b981" opacity="0.49"/>
    <rect x="0" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.46"/>
    <rect x="26" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.49"/>
    <rect x="52" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.44"/>
    <rect x="78" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.47"/>
    <rect x="104" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="130" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.45"/>
    <rect x="156" y="52" width="24" height="24" rx="2" fill="#10b981" opacity="0.48"/>
    <rect x="0" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
    <rect x="26" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.44"/>
    <rect x="52" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.47"/>
    <rect x="78" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.43"/>
    <rect x="104" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.48"/>
    <rect x="130" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.46"/>
    <rect x="156" y="78" width="24" height="24" rx="2" fill="#10b981" opacity="0.5"/>
  </g>
  <text x="460" y="170" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Uniform magnitude distribution (easy to quantize)</text>
  <!-- Properties -->
  <rect x="60" y="185" width="460" height="35" rx="6" fill="#1e293b" stroke="#94a3b8" stroke-width="1"/>
  <text x="290" y="200" text-anchor="middle" fill="#f1f5f9" font-family="system-ui" font-size="9">Key Property: R is orthogonal (R^T R = I), so ||XR|| = ||X|| -- preserves information perfectly.</text>
  <text x="290" y="214" text-anchor="middle" fill="#94a3b8" font-family="system-ui" font-size="9">Hadamard matrices enable O(n log n) fast transforms without storing R explicitly.</text>
</svg>`

};

// Export for use in build scripts or direct inclusion
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DIAGRAMS;
}
