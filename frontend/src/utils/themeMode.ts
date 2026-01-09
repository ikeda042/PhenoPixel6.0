export type ThemeMode = 'light' | 'dark'

const STORAGE_KEY = 'phenopixel.theme'

const isThemeMode = (value: string | null): value is ThemeMode =>
  value === 'light' || value === 'dark'

export const getStoredTheme = (): ThemeMode | null => {
  if (typeof window === 'undefined') return null
  const stored = window.localStorage.getItem(STORAGE_KEY)
  return isThemeMode(stored) ? stored : null
}

export const getInitialTheme = (): ThemeMode => getStoredTheme() ?? 'dark'

export const applyThemeMode = (mode: ThemeMode) => {
  if (typeof document === 'undefined') return
  const root = document.documentElement
  root.classList.toggle('dark', mode === 'dark')
  root.classList.toggle('light', mode === 'light')
  window.localStorage.setItem(STORAGE_KEY, mode)
}
