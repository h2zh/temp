// A small stand-in for the browser DOM: just enough to import index.js and landmark.js under plain node
// and look at what they render. Not a test file itself (the runner only picks up *.test.mjs).

class FakeText {
  constructor(text) {
    this.nodeType = 3;
    this.textContent = String(text);
    this.parentNode = null;
  }
}

// What innerHTML was set to. The fake never parses markup; tests read it back as a string.
class FakeMarkup extends FakeText {}

const SELECTOR_PART = /([.#][\w-]+|\[[\w-]+(?:="[^"]*")?\])/g;

function matchesCompound(node, compound) {
  const tag = compound.match(/^[a-z][\w-]*/i)?.[0];
  if (tag && node.tagName !== tag.toUpperCase()) return false;
  for (const [part] of compound.matchAll(SELECTOR_PART)) {
    if (part[0] === '.' && !node.className.split(/\s+/).includes(part.slice(1))) return false;
    if (part[0] === '#' && node.getAttribute('id') !== part.slice(1)) return false;
    if (part[0] === '[') {
      const [, name, value] = part.match(/^\[([\w-]+)(?:="([^"]*)")?\]$/);
      if (!node.hasAttribute(name) || (value !== undefined && node.getAttribute(name) !== value)) return false;
    }
  }
  return true;
}

export class FakeElement {
  constructor(tag) {
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.childNodes = [];
    this.parentNode = null;
    this.attributes = new Map();
    this.listeners = {};
    this.dataset = {};
    this.style = {};
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.innerHTMLSets = 0; // how often markup was parsed into this node
  }

  get children() {
    return this.childNodes.filter((n) => n.nodeType === 1);
  }

  get options() {
    return this.children.filter((n) => n.tagName === 'OPTION');
  }

  get firstChild() {
    return this.childNodes[0] ?? null;
  }

  get className() {
    return this.getAttribute('class') ?? '';
  }

  set className(value) {
    this.setAttribute('class', value);
  }

  get textContent() {
    return this.childNodes.map((n) => n.textContent).join('');
  }

  set textContent(value) {
    this.replaceChildren(String(value ?? ''));
  }

  get innerHTML() {
    return this.childNodes.map((n) => (n instanceof FakeMarkup ? n.textContent : n.textContent)).join('');
  }

  set innerHTML(value) {
    this.innerHTMLSets++;
    this.replaceChildren();
    if (value) this.appendChild(new FakeMarkup(value));
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.has(name) ? this.attributes.get(name) : null;
  }

  hasAttribute(name) {
    return this.attributes.has(name);
  }

  removeAttribute(name) {
    this.attributes.delete(name);
  }

  append(...nodes) {
    for (const n of nodes) this.appendChild(typeof n === 'string' ? new FakeText(n) : n);
  }

  prepend(...nodes) {
    const rest = this.childNodes;
    this.childNodes = [];
    this.append(...nodes);
    for (const n of rest) this.appendChild(n);
  }

  appendChild(node) {
    node.parentNode?.removeChild(node);
    node.parentNode = this;
    this.childNodes.push(node);
    return node;
  }

  removeChild(node) {
    this.childNodes = this.childNodes.filter((n) => n !== node);
    node.parentNode = null;
    return node;
  }

  hasChildNodes() {
    return this.childNodes.length > 0;
  }

  replaceChildren(...nodes) {
    for (const n of this.childNodes) n.parentNode = null;
    this.childNodes = [];
    this.append(...nodes);
  }

  remove() {
    this.parentNode?.removeChild(this);
  }

  focus() {}

  addEventListener(type, fn) {
    (this.listeners[type] ??= []).push(fn);
  }

  removeEventListener(type, fn) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn);
  }

  dispatch(type, init = {}) {
    const event = { type, target: this, preventDefault() {}, ...init };
    for (const fn of this.listeners[type] ?? []) fn(event);
    return event;
  }

  click() {
    this.dispatch('click');
  }

  *descendants() {
    for (const n of this.children) {
      yield n;
      yield* n.descendants();
    }
  }

  /** Compound selectors (tag, .class, #id, [attr="v"]) joined by spaces (descendant) or commas. */
  querySelectorAll(selector) {
    const groups = selector.split(',').map((s) => s.trim().split(/\s+/));
    return [...this.descendants()].filter((node) =>
      groups.some((parts) => {
        if (!matchesCompound(node, parts.at(-1))) return false;
        let i = parts.length - 2;
        for (let a = node.parentNode; a && a !== this && i >= 0; a = a.parentNode) if (matchesCompound(a, parts[i])) i--;
        return i < 0;
      }),
    );
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

/**
 * Installs document/window/location globals. getElementById hands out one element per id, created on
 * first use, so the pages run without their HTML. Returns the document; restore() undoes everything.
 */
export function installDom({ search = '' } = {}) {
  const saved = { document: globalThis.document, window: globalThis.window, location: globalThis.location };
  const byId = new Map();
  const head = new FakeElement('head');
  const body = new FakeElement('body');
  const doc = {
    head,
    body,
    title: '',
    visibilityState: 'visible',
    listeners: {},
    createElement: (tag) => new FakeElement(tag),
    getElementById(id) {
      if (!byId.has(id)) {
        const node = new FakeElement('div');
        node.setAttribute('id', id);
        body.appendChild(node);
        byId.set(id, node);
      }
      return byId.get(id);
    },
    addEventListener(type, fn) {
      (this.listeners[type] ??= []).push(fn);
    },
    restore() {
      for (const [key, value] of Object.entries(saved)) {
        if (value === undefined) delete globalThis[key];
        else globalThis[key] = value;
      }
    },
  };
  globalThis.document = doc;
  globalThis.window = globalThis;
  globalThis.location = { search };
  return doc;
}

/** Text of a node with whitespace collapsed, like innerText for these tests. */
export function text(node) {
  return node.textContent.replace(/\s+/g, ' ').trim();
}

/** Lets pending promise callbacks (fetch mocks, awaits) run. */
export async function settle(rounds = 20) {
  for (let i = 0; i < rounds; i++) await new Promise((resolve) => setImmediate(resolve));
}
