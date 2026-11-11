/**
 * Load `@molcrafts/molplot` `<molplot-chart>` for documentation pages.
 *
 * Prefer the copy staged from the local monorepo
 * (`../molplot/core/dist` → `assets/molplot/`) so fence charts use the
 * latest layout / fontScale behaviour. Fall back to jsDelivr when no
 * staged build exists.
 */
const packageBundle = new URL("../molplot/elements.js", import.meta.url);

try {
  await import(packageBundle.href);
} catch (packageError) {
  console.info(
    "MolPlot elements bundle is not staged under assets/molplot; " +
      "loading @molcrafts/molplot from the jsDelivr npm CDN.",
    packageError,
  );
  await import(
    "https://cdn.jsdelivr.net/npm/@molcrafts/molplot@0.1.6/dist/elements.js"
  );
}
