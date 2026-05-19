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

    sidebar: {
      "/LLM": [
        {
          text: "LLM",
          items: [
            { text: "概览", link: "/LLM/overview" },
            { text: "分词与Embedding", link: "/LLM/tokenizer" },
            { text: "注意力机制", link: "/LLM/attention" },
            { text: "GPT框架", link: "/LLM/gpt_structure" },
            { text: "预训练", link: "/LLM/pretrain" },
            { text: "微调", link: "/LLM/fine_tuning" },
            { text: "强化学习", link: "/LLM/reinforcement_learning" },
          ],
        },
      ],
      "/RL": [
        {
          text: "强化学习",
          items: [{ text: "概览", link: "/RL/overview" }],
        },
      ],
      "/blog": [
        {
          text: "个人博客",
          items: [
            { text: "目录", link: "/blog" },
            {
              text: "SDD: 规约驱动开发",
              link: "/blog/sdd/sdd_overview",
              items: [
                {
                  text: "SDD：个人项目开发实战",
                  link: "/blog/sdd/sdd_develop",
                },
              ],
            },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/vuejs/vitepress" },
    ],
  },
  markdown: {
    math: true,
  },
});
