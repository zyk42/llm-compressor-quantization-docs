const fs = require('fs');
const path = require('path');

const DOCS_DIR = __dirname;
const HTML_DIR = path.join(DOCS_DIR, 'html');

const PAGES = [
  { file: "00-overview.md", html: "index.html", short: "总览", num: "00" },
  { file: "01-architecture.md", html: "01-architecture.html", short: "架构", num: "01" },
  { file: "02-quantization-fundamentals.md", html: "02-quantization-fundamentals.html", short: "基础理论", num: "02" },
  { file: "03-rtn-quantization.md", html: "03-rtn-quantization.html", short: "RTN", num: "03" },
  { file: "04-gptq.md", html: "04-gptq.html", short: "GPTQ", num: "04" },
  { file: "05-awq.md", html: "05-awq.html", short: "AWQ", num: "05" },
  { file: "06-autoround.md", html: "06-autoround.html", short: "AutoRound", num: "06" },
  { file: "07-smoothquant.md", html: "07-smoothquant.html", short: "SmoothQuant", num: "07" },
  { file: "08-rotation-quantization.md", html: "08-rotation-quantization.html", short: "旋转量化", num: "08" },
  { file: "09-imatrix.md", html: "09-imatrix.html", short: "IMatrix", num: "09" },
  { file: "10-quantization-formats.md", html: "10-quantization-formats.html", short: "格式", num: "10" },
  { file: "11-kv-cache-quantization.md", html: "11-kv-cache-quantization.html", short: "KV Cache", num: "11" },
  { file: "12-advanced-practice.md", html: "12-advanced-practice.html", short: "高级实践", num: "12" },
  { file: "13-troubleshooting.md", html: "13-troubleshooting.html", short: "排查", num: "13" },
];

function escapeHtml(t) { return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function inlineFormat(text) {
  text = text.replace(/\$\$(.+?)\$\$/g, '<div class="formula">\\[$1\\]</div>');
  text = text.replace(/\$(.+?)\$/g, '\\($1\\)');
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return text;
}

function makeAnchor(t) { return t.replace(/[^\w\s\u4e00-\u9fff-]/g,'').trim().toLowerCase().replace(/\s+/g,'-').substring(0,50); }

function mdToInteractiveHtml(mdText) {
  const lines = mdText.split('\n');
  const out = [];
  let inCode = false, inTable = false, inList = false, listType = null;
  let sectionCount = 0;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Code blocks
    if (line.startsWith('```')) {
      if (inCode) { out.push('</code></pre>'); inCode = false; }
      else { const lang = line.slice(3).trim()||'text'; out.push(`<pre data-lang="${lang}"><code>`); inCode = true; }
      i++; continue;
    }
    if (inCode) { out.push(escapeHtml(line)); i++; continue; }

    // Empty
    if (!line.trim()) {
      if (inList) { out.push(`</${listType}>`); inList = false; }
      if (inTable) { out.push('</tbody></table></div>'); inTable = false; }
      i++; continue;
    }

    // Tables
    if (line.includes('|') && line.trim().startsWith('|')) {
      const cells = line.trim().split('|').slice(1,-1).map(c=>c.trim());
      if (!inTable) {
        inTable = true;
        out.push('<div class="table-wrapper reveal"><table><thead><tr>');
        cells.forEach(c => out.push(`<th>${inlineFormat(c)}</th>`));
        out.push('</tr></thead><tbody>');
        i++;
        if (i < lines.length && /^\|[\s\-:|]+\|/.test(lines[i].trim())) i++;
        continue;
      } else {
        out.push('<tr>');
        cells.forEach(c => out.push(`<td>${inlineFormat(c)}</td>`));
        out.push('</tr>');
        i++; continue;
      }
    }
    if (inTable && !line.includes('|')) { out.push('</tbody></table></div>'); inTable = false; }

    // H1 - Hero section
    if (line.startsWith('# ') && !line.startsWith('## ')) {
      // Skip - handled in template hero
      i++; continue;
    }

    // H2 - Major sections with numbered headers
    if (line.startsWith('## ')) {
      if (inList) { out.push(`</${listType}>`); inList = false; }
      sectionCount++;
      const text = line.slice(3);
      const anchor = makeAnchor(text);
      out.push(`<div class="section reveal" id="${anchor}">`);
      out.push(`<div class="section-header">`);
      out.push(`<div class="section-number">${sectionCount}</div>`);
      out.push(`<h2 class="section-title">${inlineFormat(text)}</h2>`);
      out.push(`</div>`);
      // Close previous section if exists
      if (sectionCount > 1) {
        // Insert closing div before this section's opening
        const lastSectionIdx = out.lastIndexOf('<div class="section reveal"', out.length - 6);
      }
      i++; continue;
    }

    // H3 - Sub sections
    if (line.startsWith('### ')) {
      if (inList) { out.push(`</${listType}>`); inList = false; }
      const text = line.slice(4);
      out.push(`<h3 id="${makeAnchor(text)}" class="reveal">${inlineFormat(text)}</h3>`);
      i++; continue;
    }

    // H4
    if (line.startsWith('#### ')) {
      if (inList) { out.push(`</${listType}>`); inList = false; }
      out.push(`<h4>${inlineFormat(line.slice(5))}</h4>`);
      i++; continue;
    }

    // Horizontal rule -> section closer
    if (line.match(/^---+$/)) {
      out.push('</div>'); // close section
      i++; continue;
    }

    // Blockquotes -> Callout boxes
    if (line.startsWith('> ')) {
      if (inList) { out.push(`</${listType}>`); inList = false; }
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith('> ')) { quoteLines.push(lines[i].slice(2)); i++; }
      const content = inlineFormat(quoteLines.join(' '));
      const isWarning = content.includes('注意') || content.includes('警告') || content.includes('注：');
      const type = isWarning ? 'warning' : 'info';
      const icon = isWarning ? '⚠️' : '💡';
      out.push(`<div class="callout ${type} reveal"><span class="callout-icon">${icon}</span><div class="callout-content"><p class="callout-text">${content}</p></div></div>`);
      continue;
    }

    // Unordered list
    if (/^[-*] /.test(line)) {
      if (!inList || listType !== 'ul') {
        if (inList) out.push(`</${listType}>`);
        out.push('<ul>'); inList = true; listType = 'ul';
      }
      out.push(`<li>${inlineFormat(line.slice(2))}</li>`);
      i++; continue;
    }

    // Ordered list
    if (/^\d+\. /.test(line)) {
      if (!inList || listType !== 'ol') {
        if (inList) out.push(`</${listType}>`);
        out.push('<ol>'); inList = true; listType = 'ol';
      }
      out.push(`<li>${inlineFormat(line.replace(/^\d+\. /,''))}</li>`);
      i++; continue;
    }

    if (inList) { out.push(`</${listType}>`); inList = false; listType = null; }

    // Paragraph
    out.push(`<p>${inlineFormat(line)}</p>`);
    i++;
  }

  if (inList) out.push(`</${listType}>`);
  if (inTable) out.push('</tbody></table></div>');
  if (inCode) out.push('</code></pre>');

  return out.join('\n');
}

function extractH2s(mdText) {
  const headings = [];
  for (const line of mdText.split('\n')) {
    const m = line.match(/^## (.+)/);
    if (m) headings.push({ anchor: makeAnchor(m[1]), title: m[1] });
  }
  return headings;
}

function buildNav(currentHtml) {
  return PAGES.map(p => {
    const cls = p.html === currentHtml ? ' class="active"' : '';
    return `<a href="${p.html}"${cls}>${p.num} ${p.short}</a>`;
  }).join('\n        ');
}

function buildPageNav(idx) {
  let prev = '', next = '';
  if (idx > 0) {
    const p = PAGES[idx-1];
    prev = `<a href="${p.html}"><span class="nav-label">← 上一篇</span><span class="nav-title">${p.short}</span></a>`;
  }
  if (idx < PAGES.length - 1) {
    const p = PAGES[idx+1];
    next = `<a href="${p.html}" style="text-align:right"><span class="nav-label">下一篇 →</span><span class="nav-title">${p.short}</span></a>`;
  }
  return `<div class="page-nav">${prev}${next}</div>`;
}

const TEMPLATE = (title, subtitle, num, nav, body, pageNav) => `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - LLM Compressor 量化文档</title>
  <link rel="stylesheet" href="assets/style.css">
  <script>
    MathJax = {
      tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] },
      options: { skipHtmlTags: ['code','pre'] }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
  <nav class="topnav">
    <div class="topnav-brand">LLM Compressor Docs</div>
    <div class="topnav-links">
      ${nav}
    </div>
  </nav>

  <header class="hero">
    <span class="chapter-badge">Chapter ${num}</span>
    <h1>${title}</h1>
    <p class="subtitle">${subtitle}</p>
  </header>

  <div class="container">
    <div class="content">
${body}
    </div>
${pageNav}
  </div>

  <script src="assets/main.js"></script>
</body>
</html>`;

// Subtitles for each page
const SUBTITLES = {
  "00-overview.md": "快速了解 LLM Compressor 的量化能力与使用方式",
  "01-architecture.md": "深入理解 Modifier、Recipe、Pipeline、Observer 系统",
  "02-quantization-fundamentals.md": "量化数学基础：公式、误差分析与校准原理",
  "03-rtn-quantization.md": "最简单的量化方法——FP8 Dynamic / FP8 Block / W4A16 全面解析",
  "04-gptq.md": "基于 Hessian 矩阵的最优权重量化算法",
  "05-awq.md": "激活感知的通道缩放保护量化精度",
  "06-autoround.md": "SignSGD 优化舍入决策，逐层最小化输出误差",
  "07-smoothquant.md": "通道级等效变换，将激活离群值平滑转移到权重",
  "08-rotation-quantization.md": "正交旋转降低权重不相干性，提升量化精度",
  "09-imatrix.md": "基于 E[x²] 二阶矩的通道重要性加权校准",
  "10-quantization-formats.md": "INT4/8、FP8、NVFP4、MXFP4/8 位级结构与编码详解",
  "11-kv-cache-quantization.md": "推理时 KV 缓存 FP8 量化，序列长度翻倍",
  "12-advanced-practice.md": "大模型量化、分布式、多模态、MoE 与组合配方",
  "13-troubleshooting.md": "OOM、精度下降、vLLM 部署等常见问题排查",
};

// Main build
fs.mkdirSync(path.join(HTML_DIR, 'assets'), { recursive: true });

for (let idx = 0; idx < PAGES.length; idx++) {
  const page = PAGES[idx];
  const mdPath = path.join(DOCS_DIR, page.file);
  const htmlPath = path.join(HTML_DIR, page.html);

  const mdText = fs.readFileSync(mdPath, 'utf-8');
  const titleMatch = mdText.match(/^# (.+)$/m);
  const title = titleMatch ? titleMatch[1] : page.short;
  const subtitle = SUBTITLES[page.file] || '';

  const nav = buildNav(page.html);
  const body = mdToInteractiveHtml(mdText);
  const pageNav = buildPageNav(idx);

  const html = TEMPLATE(title, subtitle, page.num, nav, body, pageNav);
  fs.writeFileSync(htmlPath, html, 'utf-8');
  console.log(`  ✓ ${page.html}`);
}

console.log(`\n✅ Done! ${PAGES.length} interactive HTML pages built.`);
