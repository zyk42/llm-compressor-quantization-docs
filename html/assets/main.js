// ========== Interactive Documentation JS ==========

document.addEventListener('DOMContentLoaded', function() {

  // === Scroll Reveal Animation ===
  const reveals = document.querySelectorAll('.reveal');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
  reveals.forEach(el => observer.observe(el));

  // === Tab Switching ===
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      const tabGroup = this.closest('.tabs');
      tabGroup.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      tabGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      this.classList.add('active');
      const target = tabGroup.querySelector('#' + this.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // === Accordion ===
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', function() {
      const item = this.closest('.accordion-item');
      const wasOpen = item.classList.contains('open');
      // Close siblings
      item.closest('.accordion').querySelectorAll('.accordion-item').forEach(i => i.classList.remove('open'));
      // Toggle current
      if (!wasOpen) item.classList.add('open');
    });
  });

  // === Active nav link ===
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.topnav-links a').forEach(link => {
    if (link.getAttribute('href') === currentPage) {
      link.classList.add('active');
    }
  });

  // === Navbar scroll effect ===
  let lastScroll = 0;
  const topnav = document.querySelector('.topnav');
  window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    if (currentScroll > 100) {
      topnav.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
    } else {
      topnav.style.boxShadow = 'none';
    }
    lastScroll = currentScroll;
  });

  // === Animated counters ===
  document.querySelectorAll('.metric-value[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count);
    const suffix = el.dataset.suffix || '';
    let current = 0;
    const increment = target / 40;
    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        el.textContent = target + suffix;
        clearInterval(timer);
      } else {
        el.textContent = Math.floor(current) + suffix;
      }
    }, 30);
  });

  // === Copy code button ===
  document.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.textContent = 'Copy';
    btn.style.cssText = 'position:absolute;top:8px;right:8px;padding:4px 10px;border-radius:4px;border:1px solid #475569;background:#1e293b;color:#94a3b8;font-size:11px;cursor:pointer;opacity:0;transition:opacity 0.2s;';
    pre.style.position = 'relative';
    pre.appendChild(btn);
    pre.addEventListener('mouseenter', () => btn.style.opacity = '1');
    pre.addEventListener('mouseleave', () => btn.style.opacity = '0');
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.textContent : pre.textContent);
      btn.textContent = 'Copied!';
      btn.style.color = '#6ee7b7';
      setTimeout(() => { btn.textContent = 'Copy'; btn.style.color = '#94a3b8'; }, 2000);
    });
  });

  // === Progress bar animation ===
  document.querySelectorAll('.progress-fill').forEach(bar => {
    const width = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = width; }, 300);
  });

});
