import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'PartiNet',
  tagline: 'Dynamic, adaptive, high performance particle picking',
  favicon: 'img/gears-gear-svgrepo-com.svg',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  url: 'https://wehi-researchcomputing.github.io',
  baseUrl: '/',
  organizationName: 'WEHI-ResearchComputing',
  projectName: 'PartiNet',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'PartiNet',
      logo: {
        alt: 'My Site Logo',
        src: 'img/gears-gear-svgrepo-com.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Tutorial',
        },
        {
          href: 'https://github.com/WEHI-ResearchComputing/PartiNet',
          label: 'GitHub',
          position: 'right',
        },
                {
          href: 'https://huggingface.co/MihinP/PartiNet',
          label: 'Model Weights',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Important Pages',
          items: [
            {
              label: 'Tutorial',
              to: '/docs/intro',
            },
            {
              label: 'Installation',
              to: '/docs/installation',
            },
          ],
        },
        {
          title: 'More',
          items: [
            // Removed links to pages that don't exist in this repo.
            // If you re-add pre-print/publication pages later, restore them here.
            {
              label: 'GitHub',
              href: 'https://github.com/WEHI-ResearchComputing/PartiNet',
            },
                        {
              label: 'Model Weights',
              href: 'https://huggingface.co/MihinP/PartiNet',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} The Walter and Eliza Hall Institute. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
