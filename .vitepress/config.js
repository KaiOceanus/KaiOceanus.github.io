import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'My Blog',
  description: 'A personal blog built with VitePress',
  base: '/',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Posts', link: '/posts/' }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/KaiOceanus' }
    ]
  }
})
