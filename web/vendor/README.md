# Third-party libraries served by the site

The pages load no script and no style from another origin (D34): the server's Content-Security-Policy
(`crowd3d/csp.py`) allows none, so the libraries the pages use are kept here and served by the site itself.
Nothing in this folder is edited by hand. `scripts/vendor_web_libs.py` writes it from the packages' npm tarballs.

| Package | Version | Licence | Tarball | Tarball integrity (the registry's `dist.integrity`) |
|---|---|---|---|---|
| three | 0.160.0 (r160) | MIT, `three/LICENSE` | `https://registry.npmjs.org/three/-/three-0.160.0.tgz` | `sha512-DLU8lc0zNIPkM7rH5/e1Ks1Z8tWCGRq6g8mPowdDJpw1CFBJMU7UoJjC6PefXW7z//SSl0b2+GCw14LB+uDhng==` |
| leaflet | 1.9.4 | BSD 2-Clause, `leaflet/LICENSE` | `https://registry.npmjs.org/leaflet/-/leaflet-1.9.4.tgz` | `sha512-nxS1ynzJOmOlHp+iL3FyWqK89GtNL8U8rvlMOsQdTTssxZwCXh8N2NB3GDQOL+YR3XnWyZAxwQixURb+FA74PA==` |

Both were taken from the npm registry on 2026-11-11 (`npm pack`), and each tarball's sha512 was compared with
`npm view <name>@<version> dist.integrity`.

## What is kept

- `three/build/three.module.js` and the eight `three/examples/jsm` files the pages import, directly or through
  another of them: `controls/OrbitControls.js`, `lines/LineMaterial.js`, `lines/LineSegments2.js`,
  `lines/LineSegmentsGeometry.js`, `loaders/GLTFLoader.js`, `loaders/PLYLoader.js`,
  `renderers/CSS2DRenderer.js`, `utils/BufferGeometryUtils.js`. The folder keeps the package's layout: the import
  maps of `web/landmark.html` and `web/stylized.html` point `three` at the build and `three/addons/` at
  `examples/jsm/`, and the addons import each other by relative paths.
- `leaflet/leaflet.js`, `leaflet/leaflet.css` and `leaflet/images/` (the package's `dist/`, without its source
  builds and source maps). `web/index.js` loads the script and the stylesheet. `leaflet.js` names `leaflet.js.map`
  in its last line: a browser's developer tools ask for it and get a 404; the pages never do.
- The files are kept byte for byte as the tarballs hold them. Leaflet's stylesheet and LICENSE have CR LF line
  ends: `.gitattributes` at the repository root marks this folder `-text`, so git never rewrites them.
- `SHA256SUMS` lists every file above with its sha256. To check the folder: `cd web/vendor && sha256sum -c
  SHA256SUMS`, or `python scripts/vendor_web_libs.py --check`. `tests/test_csp.py` does the same, and checks that
  every import in `three/` resolves inside the folder and that this table names the pinned versions.

## How to update a library

1. Read the new tarball's integrity: `npm view <name>@<version> dist.integrity`.
2. In `scripts/vendor_web_libs.py`, change the package's `version` and `integrity` in `PACKAGES`; change its file
   list too if the pages import other files (for three: every `three/addons/...` import in `web/*.js`, plus what
   those files import).
3. Run `python scripts/vendor_web_libs.py`. It downloads the tarballs from the npm registry (or takes them from
   `--tarballs DIR`), refuses one whose sha512 is not the pinned one, replaces `three/` and `leaflet/` and rewrites
   `SHA256SUMS`.
4. Update the table above, run the tests (`python -m pytest -q -m "not slow"`, `node --test web/tests/*.test.mjs`)
   and look at every page in a browser with the console open: three.js changes its addons between releases, and a
   library that starts writing inline styles or loading from elsewhere would be stopped by the
   Content-Security-Policy.
