const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const markdownIt = require("markdown-it");
const markdownItFootnote = require("markdown-it-footnote");
const markdownItAttrs = require("markdown-it-attrs");
const markdownItKatex = require("markdown-it-katex");
const yaml = require("js-yaml");

module.exports = function (eleventyConfig) {
  // YAML data file support
  eleventyConfig.addDataExtension("yaml,yml", (contents) => yaml.load(contents));

  // Plugins
  eleventyConfig.addPlugin(syntaxHighlight);

  // Markdown config
  const md = markdownIt({
    html: true,
    linkify: true,
    typographer: true,
  })
    .use(markdownItFootnote)
    .use(markdownItAttrs)
    .use(markdownItKatex);

  // Override footnote_ref to render as plain number (no brackets).
  // Brackets are reserved for citations ({% cite N %}), keeping the two
  // visually distinct: footnotes → ¹  citations → [1]
  md.renderer.rules.footnote_ref = function (tokens, idx) {
    const n = tokens[idx].meta.id + 1;
    const subId = tokens[idx].meta.subId;
    const suffix = subId > 0 ? `:${subId}` : "";
    return `<sup class="footnote-ref"><a href="#fn${n}" id="fnref${n}${suffix}">${n}</a></sup>`;
  };

  eleventyConfig.setLibrary("md", md);

  // Pass-through copies
  eleventyConfig.addPassthroughCopy("src/assets");
  eleventyConfig.addPassthroughCopy("src/images");
  eleventyConfig.addPassthroughCopy("src/data");
  eleventyConfig.addPassthroughCopy("legacy");
  eleventyConfig.addPassthroughCopy({ "cv": "cv" }); // root-level cv/ → _site/cv/
  eleventyConfig.addPassthroughCopy({ "demos": "demos" }); // root-level demos/ → _site/demos/

  // Collections
  eleventyConfig.addCollection("posts", function (collectionApi) {
    return collectionApi
      .getFilteredByGlob("src/blog/posts/*.md")
      .sort((a, b) => b.date - a.date);
  });

  eleventyConfig.addCollection("tagList", function (collectionApi) {
    const tagSet = new Set();
    collectionApi.getFilteredByGlob("src/blog/posts/*.md").forEach((item) => {
      (item.data.tags || []).forEach((tag) => tagSet.add(tag));
    });
    return [...tagSet].sort();
  });

  // Filters
  eleventyConfig.addFilter("dateDisplay", function (date) {
    return new Date(date).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  });

  eleventyConfig.addFilter("dateYear", function (date) {
    return new Date(date).getFullYear();
  });

  eleventyConfig.addFilter("isoDate", function (date) {
    return new Date(date).toISOString().split("T")[0];
  });

  eleventyConfig.addFilter("limit", function (arr, n) {
    if (!arr || !Array.isArray(arr)) return [];
    return arr.slice(0, n);
  });

  // Merge highlights + other, remove under-submission, sort by year desc
  eleventyConfig.addFilter("pubsVisible", function (publications) {
    const all = [...(publications.highlights || []), ...(publications.other || [])];
    return all
      .filter((p) => p.status !== "under-submission")
      .sort((a, b) => b.year - a.year);
  });

  // Shortcodes
  // Inline citation marker: {% cite N %} → superscript [N] linking to #ref-N
  // Note: brackets are added by CSS ::before/::after to avoid markdown-it
  // mis-parsing [N] inside HTML as a reference-style link.
  eleventyConfig.addShortcode("cite", function (num) {
    return `<sup class="cite-marker"><a href="#ref-${num}" id="citeref-${num}" class="cite-link">${num}</a></sup>`;
  });

  eleventyConfig.addShortcode("demo", function (scriptPath, caption) {
    const id = scriptPath.replace(/[^a-z0-9]/gi, "_");
    return `<div class="demo-container">
  <canvas id="${id}" class="demo-canvas"></canvas>
  <script src="${scriptPath}"></script>${caption ? `\n  <p class="demo-caption">${caption}</p>` : ""}
</div>`;
  });

  return {
    dir: {
      input: "src",
      output: "_site",
      includes: "_includes",
      data: "_data",
    },
    templateFormats: ["njk", "md", "html"],
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
  };
};
