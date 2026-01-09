import { createSystem, defaultConfig, defineConfig, defineSemanticTokens } from '@chakra-ui/react'

const semanticColors = defineSemanticTokens.colors({
  ink: {
    900: { value: { _dark: '#f8fafc', _light: '#0b0d10' } },
    800: { value: { _dark: '#e2e8f0', _light: '#111827' } },
    700: { value: { _dark: '#9aa3af', _light: '#334155' } },
  },
  sand: {
    50: { value: { _dark: '#0b0d10', _light: '#f8fafc' } },
    100: { value: { _dark: '#111318', _light: '#edf2f7' } },
    200: { value: { _dark: '#1a1d24', _light: '#d5dee9' } },
    300: { value: { _dark: '#242833', _light: '#bcc8d8' } },
  },
  tide: {
    300: { value: { _dark: '#2dd4bf', _light: '#5eead4' } },
    400: { value: { _dark: '#22b3a1', _light: '#2dd4bf' } },
    500: { value: { _dark: '#178d80', _light: '#22b3a1' } },
    600: { value: { _dark: '#0f625a', _light: '#1b9a8d' } },
  },
  violet: {
    300: { value: { _dark: '#c4b5fd', _light: '#a78bfa' } },
    400: { value: { _dark: '#a78bfa', _light: '#c4b5fd' } },
    500: { value: { _dark: '#8b5cf6', _light: '#ddd6fe' } },
    600: { value: { _dark: '#6d28d9', _light: '#ede9fe' } },
  },
})

const customConfig = defineConfig({
  theme: {
    tokens: {
      fonts: {
        heading: {
          value:
            '"Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif',
        },
        body: {
          value:
            '"Noto Sans JP", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", sans-serif',
        },
      },
      colors: {
        ink: {
          900: { value: '#f8fafc' },
          800: { value: '#e2e8f0' },
          700: { value: '#9aa3af' },
        },
        sand: {
          50: { value: '#0b0d10' },
          100: { value: '#111318' },
          200: { value: '#1a1d24' },
          300: { value: '#242833' },
        },
        tide: {
          300: { value: '#2dd4bf' },
          400: { value: '#22b3a1' },
          500: { value: '#178d80' },
          600: { value: '#0f625a' },
        },
        violet: {
          300: { value: '#c4b5fd' },
          400: { value: '#a78bfa' },
          500: { value: '#8b5cf6' },
          600: { value: '#6d28d9' },
        },
      },
    },
    semanticTokens: {
      colors: semanticColors,
    },
  },
  globalCss: {
    body: {
      bg: 'sand.50',
      color: 'ink.900',
      fontFamily: 'body',
    },
  },
})

const system = createSystem(defaultConfig, customConfig)

export default system
