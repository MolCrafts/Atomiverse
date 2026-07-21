/**
 * Load the MolVis web-component bundle for documentation pages.
 *
 * Prefer the staged package under assets/molvis-core/ (built from the monorepo).
 * Fall back to jsDelivr only when that file exists on the published package.
 */
const packageBundle = new URL("../molvis-core/elements.js", import.meta.url);

try {
  await import(packageBundle.href);
} catch (packageError) {
  console.info(
    "MolVis package bundle not staged under assets/molvis-core; " +
      "trying @molcrafts/molvis-core from jsDelivr.",
    packageError,
  );
  await import(
    "https://cdn.jsdelivr.net/npm/@molcrafts/molvis-core@latest/dist/elements.js"
  );
}
