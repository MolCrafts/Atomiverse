/**
 * Load `@molcrafts/molvis-stage-viewer` for documentation pages.
 *
 * Prefer the copy staged from the local monorepo
 * (`../molvis/stage-viewer/dist` → `assets/molvis-stage-viewer/`) so the
 * docs site exercises the latest workspace build. Fall back to jsDelivr
 * when no staged build exists.
 *
 * molpy is wasm-bindgen bundler-target only; the viewer entry
 * (`dist/main.js`) must ship a properly wired WASM module.
 */
const packageBundle = new URL(
  "../molvis-stage-viewer/main.js",
  import.meta.url,
);

try {
  await import(packageBundle.href);
} catch (packageError) {
  console.info(
    "MolVis stage-viewer bundle is not staged under assets/molvis-stage-viewer; " +
      "loading @molcrafts/molvis-stage-viewer from the jsDelivr npm CDN.",
    packageError,
  );
  await import(
    "https://cdn.jsdelivr.net/npm/@molcrafts/molvis-stage-viewer@0.3.0/dist/main.js"
  );
}
