import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// SITE_BASE lets GitHub Pages serve the site from /<repo>/.
export default defineConfig({
  base: process.env.SITE_BASE ?? "/",
  trailingSlash: "always",
  integrations: [react(), mdx()],
  markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }) },
  vite: { worker: { format: "es" } },
});
