"""Build HTML files from markdown sources."""
import re
import os

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
MD_DIR = DOCS_DIR
HTML_DIR = os.path.join(DOCS_DIR, "html")

SIDEBAR_LINKS = [
    ("index.html", "00 - 总览与快速入门"),
    ("01-architecture.html", "01 - 项目架构解析"),
    ("02-quantization-fundamentals.html", "02 - 量化基础理论"),
    ("03-rtn-quantization.html", "03 - RTN 最近舍入量化"),
    ("04-gptq.html", "04 - GPTQ 量化详解"),
    ("05-awq.html", "05 - AWQ 量化详解"),
    ("06-autoround.html", "06 - AutoRound 量化"),
    ("07-smoothquant.html", "07 - SmoothQuant 详解"),
    ("08-rotation-quantization.html", "08 - 旋转量化 (SpinQuant/QuIP)"),
    ("09-imatrix.html", "09 - IMatrix 重要性校准"),
    ("10-quantization-formats.html", "10 - 量化格式详解"),
    ("11-kv-cache-quantization.html", "11 - KV Cache 量化"),
    ("12-advanced-practice.html", "12 - 高级实践"),
    ("13-troubleshooting.html", "13 - 问题排查"),
]

MD_FILES = [
    ("00-overview.md", "index.html"),
    ("01-architecture.md", "01-architecture.html"),
    ("02-quantization-fundamentals.md", "02-quantization-fundamentals.html"),
    ("03-rtn-quantization.md", "03-rtn-quantization.html"),
    ("04-gptq.md", "04-gptq.html"),
    ("05-awq.md", "05-awq.html"),
    ("06-autoround.md", "06-autoround.html"),
    ("07-smoothquant.md", "07-smoothquant.html"),
    ("08-rotation-quantization.md", "08-rotation-quantization.html"),
    ("09-imatrix.md", "09-imatrix.html"),
    ("10-quantization-formats.md", "10-quantization-formats.html"),
    ("11-kv-cache-quantization.md", "11-kv-cache-quantization.html"),
    ("12-advanced-practice.md", "12-advanced-practice.html"),
    ("13-troubleshooting.md", "13-troubleshooting.html"),
]


def md_to_html_content(md_text):
    """Convert markdown to HTML content (simplified converter)."""
    lines = md_text.split('\n')
    html_lines = []
    in_code_block = False
    in_table = False
    in_list = False
    list_type = None
    table_header_done = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Code blocks
        if line.startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                lang = line[3:].strip()
                html_lines.append(f'<pre><code class="language-{lang}">')
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            # Escape HTML in code blocks
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html_lines.append(escaped)
            i += 1
            continue

        # Empty line
        if not line.strip():
            if in_list:
                html_lines.append(f'</{list_type}>')
                in_list = False
                list_type = None
            if in_table:
                html_lines.append('</tbody></table>')
                in_table = False
                table_header_done = False
            html_lines.append('')
            i += 1
            continue

        # Tables
        if '|' in line and line.strip().startswith('|'):
            cells = [c.strip() for c in line.strip().split('|')[1:-1]]
            if not in_table:
                in_table = True
                table_header_done = False
                html_lines.append('<table><thead><tr>')
                for cell in cells:
                    html_lines.append(f'<th>{inline_format(cell)}</th>')
                html_lines.append('</tr></thead>')
                i += 1
                # Skip separator line
                if i < len(lines) and re.match(r'\|[\s\-:|]+\|', lines[i].strip()):
                    i += 1
                html_lines.append('<tbody>')
                table_header_done = True
                continue
            else:
                html_lines.append('<tr>')
                for cell in cells:
                    html_lines.append(f'<td>{inline_format(cell)}</td>')
                html_lines.append('</tr>')
                i += 1
                continue

        # Close table if line doesn't have |
        if in_table and '|' not in line:
            html_lines.append('</tbody></table>')
            in_table = False
            table_header_done = False

        # Headers
        if line.startswith('# '):
            text = line[2:]
            anchor = make_anchor(text)
            html_lines.append(f'<h1 id="{anchor}">{inline_format(text)}</h1>')
            i += 1
            continue
        if line.startswith('## '):
            text = line[3:]
            anchor = make_anchor(text)
            html_lines.append(f'<h2 id="{anchor}">{inline_format(text)}</h2>')
            i += 1
            continue
        if line.startswith('### '):
            text = line[4:]
            anchor = make_anchor(text)
            html_lines.append(f'<h3 id="{anchor}">{inline_format(text)}</h3>')
            i += 1
            continue
        if line.startswith('#### '):
            text = line[5:]
            html_lines.append(f'<h4>{inline_format(text)}</h4>')
            i += 1
            continue

        # Blockquotes
        if line.startswith('> '):
            quote_lines = []
            while i < len(lines) and lines[i].startswith('> '):
                quote_lines.append(lines[i][2:])
                i += 1
            html_lines.append('<blockquote><p>' + inline_format(' '.join(quote_lines)) + '</p></blockquote>')
            continue

        # Unordered lists
        if re.match(r'^[-*] ', line):
            if not in_list or list_type != 'ul':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ul>')
                in_list = True
                list_type = 'ul'
            html_lines.append(f'<li>{inline_format(line[2:])}</li>')
            i += 1
            continue

        # Ordered lists
        if re.match(r'^\d+\. ', line):
            if not in_list or list_type != 'ol':
                if in_list:
                    html_lines.append(f'</{list_type}>')
                html_lines.append('<ol>')
                in_list = True
                list_type = 'ol'
            text = re.sub(r'^\d+\. ', '', line)
            html_lines.append(f'<li>{inline_format(text)}</li>')
            i += 1
            continue

        # Close list if not a list item
        if in_list:
            html_lines.append(f'</{list_type}>')
            in_list = False
            list_type = None

        # Paragraph
        html_lines.append(f'<p>{inline_format(line)}</p>')
        i += 1

    # Close any open blocks
    if in_list:
        html_lines.append(f'</{list_type}>')
    if in_table:
        html_lines.append('</tbody></table>')
    if in_code_block:
        html_lines.append('</code></pre>')

    return '\n'.join(html_lines)


def inline_format(text):
    """Apply inline formatting: bold, code, italic, links."""
    # Inline math: $...$
    text = re.sub(r'\$\$(.+?)\$\$', r'<span class="formula-block">\\[\1\\]</span>', text)
    text = re.sub(r'\$(.+?)\$', r'\\(\1\\)', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def make_anchor(text):
    """Create URL-friendly anchor from heading text."""
    text = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', text)
    text = text.strip().lower().replace(' ', '-')
    return text[:50]


def extract_toc(md_text):
    """Extract h2 headings for table of contents."""
    headings = []
    for line in md_text.split('\n'):
        if line.startswith('## '):
            text = line[3:]
            anchor = make_anchor(text)
            headings.append((anchor, text))
    return headings


def build_sidebar(current_file):
    """Build sidebar HTML."""
    links = []
    for href, title in SIDEBAR_LINKS:
        active = ' class="active"' if href == current_file else ''
        links.append(f'      <a href="{href}"{active}>{title}</a>')
    return '\n'.join(links)


def build_page_nav(current_idx):
    """Build previous/next navigation."""
    prev_html = '<span></span>'
    next_html = '<span></span>'

    if current_idx > 0:
        prev_href, prev_title = SIDEBAR_LINKS[current_idx - 1]
        prev_html = f'<a href="{prev_href}"><span><span class="label">上一篇</span>&larr; {prev_title}</span></a>'

    if current_idx < len(SIDEBAR_LINKS) - 1:
        next_href, next_title = SIDEBAR_LINKS[current_idx + 1]
        next_html = f'<a href="{next_href}"><span><span class="label">下一篇</span>{next_title} &rarr;</span></a>'

    return f'<div class="page-nav">{prev_html}{next_html}</div>'


def build_toc_html(headings):
    """Build table of contents HTML."""
    if not headings:
        return ''
    items = '\n'.join(f'          <li><a href="#{anchor}">{title}</a></li>' for anchor, title in headings)
    return f'''      <div class="toc">
        <div class="toc-title">目录</div>
        <ol>
{items}
        </ol>
      </div>'''


TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} - LLM Compressor 量化文档</title>
  <link rel="stylesheet" href="assets/style.css">
  <script>
    MathJax = {{
      tex: {{ inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']] }},
      options: {{ skipHtmlTags: ['code', 'pre'] }}
    }};
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
{sidebar}
    </nav>
  </aside>

  <main class="main">
    <div class="content">
{toc}
{body}

{page_nav}
    </div>
  </main>
  <script src="assets/main.js"></script>
</body>
</html>'''


def main():
    os.makedirs(HTML_DIR, exist_ok=True)

    for idx, (md_file, html_file) in enumerate(MD_FILES):
        if html_file == "index.html":
            continue  # Already created manually

        md_path = os.path.join(MD_DIR, md_file)
        html_path = os.path.join(HTML_DIR, html_file)

        with open(md_path, 'r', encoding='utf-8') as f:
            md_text = f.read()

        # Extract title (first h1)
        title_match = re.search(r'^# (.+)$', md_text, re.MULTILINE)
        title = title_match.group(1) if title_match else md_file

        # Build components
        toc = build_toc_html(extract_toc(md_text))
        body = md_to_html_content(md_text)
        sidebar = build_sidebar(html_file)
        page_nav = build_page_nav(idx)

        html_content = TEMPLATE.format(
            title=title,
            sidebar=sidebar,
            toc=toc,
            body=body,
            page_nav=page_nav,
        )

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"  Built: {html_file}")

    print(f"\nDone! {len(MD_FILES)} HTML files generated in {HTML_DIR}")


if __name__ == '__main__':
    main()
