// Small DOM and formatting helpers shared by the pages. Text always goes in via
// textContent, so names, user ids and file names from the API are never parsed as HTML.

/**
 * el('a', {class: 'x', href: '#', text: 'hi', onclick: fn, dataset: {k: 'v'}}, ...children)
 * Attribute values of null/undefined/false are skipped; true sets an empty attribute.
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value; // only for markup built locally (compass SVG)
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  return node;
}

export function fmtPercent(p) {
  return `${Math.round(Math.max(0, Math.min(1, p || 0)) * 100)}%`;
}

export function fmtMeters(m) {
  if (m === null || m === undefined) return '—';
  return m < 10 ? `${m.toFixed(1)} 米` : `${Math.round(m)} 米`;
}

export function fmtDeg(d) {
  return d === null || d === undefined ? '—' : `${Math.round(d) % 360}°`;
}

export function fmtCount(n) {
  return (n ?? 0).toLocaleString('zh-CN');
}

/** Unix seconds -> "刚刚" / "5 分钟前" / "3 小时前" / "10月21日 14:05". */
export function fmtTime(unix, now = Date.now() / 1000) {
  if (!unix) return '—';
  const s = Math.max(0, now - unix);
  if (s < 60) return '刚刚';
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  const d = new Date(unix * 1000);
  const hh = String(d.getHours()).padStart(2, '0');
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${d.getMonth() + 1}月${d.getDate()}日 ${hh}:${mm}`;
}

export function progressBar(progress, label) {
  const pct = Math.round(Math.max(0, Math.min(1, progress || 0)) * 100);
  const bar = el('div', {
    class: 'progress',
    role: 'progressbar',
    'aria-valuemin': 0,
    'aria-valuemax': 100,
    'aria-valuenow': pct,
    'aria-label': label,
  });
  bar.append(el('div', { class: 'progress-fill', style: `width: ${pct}%` }));
  return bar;
}

/** Replace a container's children. */
export function fill(container, ...children) {
  container.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}
