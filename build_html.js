const fs = require('fs');
const path = require('path');

const DOCS_DIR = __dirname;
const HTML_DIR = path.join(DOCS_DIR, 'html');

const SIDEBAR_LINKS = [
  ["index.html", "00 - 总览与快速入门"],
  ["01-architecture.html", "01 - 项目架构解析"],
  ["02-quantization-fundamentals.html", "02 - 量化基础理论"],
  ["03-rtn-quantization.html", "03 - RTN 最近舍入量化"],
  ["04-gptq.html", "04 - GPTQ 量化详解"],
  ["05-awq.html", "05 - AWQ 量化详解"],
  ["06-autoround.html", "06 - AutoRound 量化"],
  ["07-smoothquant.html", "07 - SmoothQuant 详解"],
  ["08-rotation-quantization.html", "08 - 旋转量化 (SpinQuant/QuIP)"],
  ["09-imatrix.html", "09 - IMatrix 重要性校准"],
  ["10-quantization-formats.html", "10 - 量化格式详解"],
  ["11-kv-cache-quantization.html", "11 - KV Cache 量化"],
  ["12-advanced-practice.html", "12 - 高级实践"],
  ["13-troubleshooting.html", "13 - 问题排查"],
];

const MD_FILES = [
  ["00-overview.md", "index.html"],
  ["01-architecture.md", "01-architecture.html"],
  ["02-quantization-fundamentals.md", "02-quantization-fundamentals.html"],
  ["03-rtn-quantization.md", "03-rtn-quantization.html"],
  ["04-gptq.md", "04-gptq.html"],
  ["05-awq.md", "05-awq.html"],
  ["06-autoround.md", "06-autoround.html"],
  ["07-smoothquant.md", "07-smoothquant.html"],
  ["08-rotation-quantization.md", "08-rotation-quantization.html"],
  ["09-imatrix.md", "09-imatrix.html"],
  ["10-quantization-formats.md", "10-quantization-formats.html"],
  ["11-kv-cache-quantization.md", "11-kv-cache-quantization.html"],
  ["12-advanced-practice.md", "12-advanced-practice.html"],
  ["13-troubleshooting.md", "13-troubleshooting.html"],
];

function escapeHtml(text) {
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function inlineFormat(text) {
  // Math: $$...$$ (display)
  text = text.replace(/\$\$(.+?)\$\$/g, '<div class="formula-block">\\[$1\\]</div>');
  // Math: $...$ (inline)
  text = text.replace(/\$(.+?)\$/g, '\\($1\\)');
  // Inline code (before bold/italic to avoid conflicts)
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Links
  text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
  return text;
}

function makeAnchor(text) {
  return text.replace(/[^\w\s\u4e00-\u9fff-]/g, '').trim().toLowerCase().replace(/\s+/g, '-').substring(0, 50);
}

function mdToHtml(mdText) {
  const lines = mdText.split('\n');
  const htmlLines = [];
  let inCodeBlock = false;
  let inTable = false;
  let inList = false;
  let listType = null;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Code blocks
    if (line.startsWith('```')) {
      if (inCodeBlock) {
        htmlLines.push('</code></pre>');
        inCodeBlock = false;
      } else {
        const lang = line.slice(3).trim();
        htmlLines.push(`<pre><code class="language-${lang || 'text'}">`);
        inCodeBlock = true;
      }
      i++;
      continue;
    }

    if (inCodeBlock) {
      htmlLines.push(escapeHtml(line));
      i++;
      continue;
    }

    // Empty line
    if (!line.trim()) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; listType = null; }
      if (inTable) { htmlLines.push('</tbody></table>'); inTable = false; }
      i++;
      continue;
    }

    // Tables
    if (line.includes('|') && line.trim().startsWith('|')) {
      const cells = line.trim().split('|').slice(1, -1).map(c => c.trim());
      if (!inTable) {
        inTable = true;
        htmlLines.push('<table><thead><tr>');
        cells.forEach(c => htmlLines.push(`<th>${inlineFormat(c)}</th>`));
        htmlLines.push('</tr></thead>');
        i++;
        // Skip separator
        if (i < lines.length && /^\|[\s\-:|]+\|/.test(lines[i].trim())) i++;
        htmlLines.push('<tbody>');
        continue;
      } else {
        htmlLines.push('<tr>');
        cells.forEach(c => htmlLines.push(`<td>${inlineFormat(c)}</td>`));
        htmlLines.push('</tr>');
        i++;
        continue;
      }
    }

    if (inTable && !line.includes('|')) {
      htmlLines.push('</tbody></table>');
      inTable = false;
    }

    // Headers
    const h1Match = line.match(/^# (.+)/);
    const h2Match = line.match(/^## (.+)/);
    const h3Match = line.match(/^### (.+)/);
    const h4Match = line.match(/^#### (.+)/);

    if (h1Match) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      const text = h1Match[1];
      htmlLines.push(`<h1 id="${makeAnchor(text)}">${inlineFormat(text)}</h1>`);
      i++; continue;
    }
    if (h2Match) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      const text = h2Match[1];
      htmlLines.push(`<h2 id="${makeAnchor(text)}">${inlineFormat(text)}</h2>`);
      i++; continue;
    }
    if (h3Match) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      const text = h3Match[1];
      htmlLines.push(`<h3 id="${makeAnchor(text)}">${inlineFormat(text)}</h3>`);
      i++; continue;
    }
    if (h4Match) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      htmlLines.push(`<h4>${inlineFormat(h4Match[1])}</h4>`);
      i++; continue;
    }

    // Blockquotes
    if (line.startsWith('> ')) {
      if (inList) { htmlLines.push(`</${listType}>`); inList = false; }
      const quoteLines = [];
      while (i < lines.length && lines[i].startsWith('> ')) {
        quoteLines.push(lines[i].slice(2));
        i++;
      }
      htmlLines.push(`<blockquote><p>${inlineFormat(quoteLines.join(' '))}</p></blockquote>`);
      continue;
    }

    // Unordered list
    if (/^[-*] /.test(line)) {
      if (!inList || listType !== 'ul') {
        if (inList) htmlLines.push(`</${listType}>`);
        htmlLines.push('<ul>');
        inList = true; listType = 'ul';
      }
      htmlLines.push(`<li>${inlineFormat(line.slice(2))}</li>`);
      i++; continue;
    }

    // Ordered list
    if (/^\d+\. /.test(line)) {
      if (!inList || listType !== 'ol') {
        if (inList) htmlLines.push(`</${listType}>`);
        htmlLines.push('<ol>');
        inList = true; listType = 'ol';
      }
      const text = line.replace(/^\d+\. /, '');
      htmlLines.push(`<li>${inlineFormat(text)}</li>`);
      i++; continue;
    }

    // Close list
    if (inList) { htmlLines.push(`</${listType}>`); inList = false; listType = null; }

    // Paragraph
    htmlLines.push(`<p>${inlineFormat(line)}</p>`);
    i++;
  }

  if (inList) htmlLines.push(`</${listType}>`);
  if (inTable) htmlLines.push('</tbody></table>');
  if (inCodeBlock) htmlLines.push('</code></pre>');

  return htmlLines.join('\n');
}

function extractToc(mdText) {
  const headings = [];
  for (const line of mdText.split('\n')) {
    const match = line.match(/^## (.+)/);
    if (match) {
      headings.push([makeAnchor(match[1]), match[1]]);
    }
  }
  return headings;
}

function buildSidebar(currentFile) {
  return SIDEBAR_LINKS.map(([href, title]) => {
    const active = href === currentFile ? ' class="active"' : '';
    return `      <a href="${href}"${active}>${title}</a>`;
  }).join('\n');
}

function buildPageNav(idx) {
  let prev = '<span></span>';
  let next = '<span></span>';
  if (idx > 0) {
    const [href, title] = SIDEBAR_LINKS[idx - 1];
    prev = `<a href="${href}"><span><span class="label">上一篇</span>&larr; ${title}</span></a>`;
  }
  if (idx < SIDEBAR_LINKS.length - 1) {
    const [href, title] = SIDEBAR_LINKS[idx + 1];
    next = `<a href="${href}"><span><span class="label">下一篇</span>${title} &rarr;</span></a>`;
  }
  return `<div class="page-nav">${prev}${next}</div>`;
}

function buildTocHtml(headings) {
  if (!headings.length) return '';
  const items = headings.map(([anchor, title]) =>
    `          <li><a href="#${anchor}">${title}</a></li>`
  ).join('\n');
  return `      <div class="toc">
        <div class="toc-title">目录</div>
        <ol>
${items}
        </ol>
      </div>`;
}

const TEMPLATE = (title, sidebar, toc, body, pageNav) => `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - LLM Compressor 量化文档</title>
  <link rel="stylesheet" href="assets/style.css">
  <script>
    MathJax = {
      tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] },
      options: { skipHtmlTags: ['code', 'pre'] }
    };
  </script>
  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
  <button class="menu-toggle">&#9776;</button>
  <aside class="sidebar">
    <div class="sidebar-header">
      <h2>LLM Compressor</h2>
      <p>量化方法详解文档</p>
    </div>
    <nav>
${sidebar}
    </nav>
  </aside>

  <main class="main">
    <div class="content">
${toc}
${body}

${pageNav}
    </div>
  </main>
  <script src="assets/main.js"></script>
</body>
</html>`;

// Main
fs.mkdirSync(path.join(HTML_DIR, 'assets'), { recursive: true });

let built = 0;
for (let idx = 0; idx < MD_FILES.length; idx++) {
  const [mdFile, htmlFile] = MD_FILES[idx];

  // Skip index.html (already created)
  if (htmlFile === 'index.html') continue;

  const mdPath = path.join(DOCS_DIR, mdFile);
  const htmlPath = path.join(HTML_DIR, htmlFile);

  const mdText = fs.readFileSync(mdPath, 'utf-8');

  // Extract title
  const titleMatch = mdText.match(/^# (.+)$/m);
  const title = titleMatch ? titleMatch[1] : mdFile;

  const toc = buildTocHtml(extractToc(mdText));
  const body = mdToHtml(mdText);
  const sidebar = buildSidebar(htmlFile);
  const pageNav = buildPageNav(idx);

  const htmlContent = TEMPLATE(title, sidebar, toc, body, pageNav);
  fs.writeFileSync(htmlPath, htmlContent, 'utf-8');
  built++;
  console.log(`  Built: ${htmlFile}`);
}

console.log(`\nDone! ${built + 1} HTML files in ${HTML_DIR}`);
