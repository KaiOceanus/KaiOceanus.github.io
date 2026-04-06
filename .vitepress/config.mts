import { defineConfig } from "vitepress";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Ortega's Tech Blog",
  description: "Tech Blog",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: "Home", link: "/" },
      { text: "Examples", link: "/markdown-examples" },
    ],

    sidebar: [
      {
        text: "LLM",
        items: [
          { text: "概览", link: "/LLM/overview" },
          { text: "分词与Embedding", link: "/LLM/tokenizer" },
          { text: "Transformer机制", link: "/LLM/transformer" },
          { text: "GPT框架", link: "/LLM/gpt_structure" },
          { text: "预训练", link: "/LLM/pretrain" },
          { text: "微调", link: "/LLM/fine_tuning" },
          { text: "强化学习", link: "/LLM/reinforcement_learning" },
        ],
      },
    ],

    socialLinks: [
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
  },
  markdown: {
    math: true,
  },
});
