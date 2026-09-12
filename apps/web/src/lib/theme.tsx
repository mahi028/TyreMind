/**
 * Theme state, and a way for canvas-based charts to react to it.
 *
 * Two things make this less trivial than it looks:
 *
 * 1. **ECharts and three.js draw to canvas**, so they cannot resolve
 *    `var(--colour)`. They need concrete values, re-read whenever the theme
 *    changes. `useThemeColour` exists so a chart re-renders with the right
 *    palette instead of keeping the dark one on a white page.
 * 2. **The choice has to survive a reload**, and the first paint must already be
 *    correct — flashing dark before switching to light is worse than not having
 *    the feature.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'tyremind.theme'

interface ThemeContextValue {
  theme: Theme
  toggle: () => void
  /** Increments on every theme change, so charts can key off it and redraw. */
  revision: number
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: 'dark',
  toggle: () => undefined,
  revision: 0,
})

/** Read the stored preference, falling back to the OS setting. */
export function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'dark' || stored === 'light') return stored
  } catch {
    /* private browsing or storage disabled; fall through to the OS setting */
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() =>
    typeof window === 'undefined' ? 'dark' : initialTheme(),
  )
  const [revision, setRevision] = useState(0)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* nothing to recover from; the theme still applies this session */
    }
    // Bump after the attribute lands, so anything re-reading computed styles
    // sees the new values rather than the old ones.
    setRevision((r) => r + 1)
  }, [theme])

  const toggle = useCallback(
    () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')),
    [],
  )

  const value = useMemo(() => ({ theme, toggle, revision }), [theme, toggle, revision])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext)
}

/**
 * Concrete colours for canvas rendering, refreshed whenever the theme changes.
 *
 * Charts should read every colour from here rather than hard-coding hex values,
 * which is what previously left axis labels invisible on a light background.
 */
export function useThemeColours() {
  const { theme, revision } = useTheme()

  return useMemo(() => {
    const read = (name: string, fallback: string) => {
      if (typeof window === 'undefined') return fallback
      return (
        getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback
      )
    }
    return {
      theme,
      // Depth scale
      ground:      read('--color-ground',       '#080c0f'),
      surface:     read('--color-surface',      '#0d1418'),
      raised:      read('--color-raised',       '#141e25'),
      card:        read('--color-card',         '#111920'),
      line:        read('--color-line',         '#1e2d38'),
      lineBright:  read('--color-line-bright',  '#2b3f4e'),
      lineSubtle:  read('--color-line-subtle',  '#161f27'),
      // Type scale
      ink:         read('--color-ink',          '#e8edf0'),
      inkDim:      read('--color-ink-dim',      '#8fa4b2'),
      inkFaint:    read('--color-ink-faint',    '#556a78'),
      inkGhost:    read('--color-ink-ghost',    '#2e4050'),
      // Compounds (warm)
      soft:        read('--color-soft',         '#e8352e'),
      medium:      read('--color-medium',       '#f5c518'),
      hard:        read('--color-hard',         '#dde0e3'),
      intermediate:read('--color-intermediate', '#3ab032'),
      wet:         read('--color-wet',          '#1e6fe0'),
      // Confounders (cool)
      fuel:        read('--color-fuel',         '#41a3c2'),
      track:       read('--color-track',        '#6b85a0'),
      traffic:     read('--color-traffic',      '#9e6ec8'),
      residual:    read('--color-residual',     '#445a69'),
      // Feedback
      alert:       read('--color-alert',        '#16a34a'),
      alertDim:    read('--color-alert-dim',    '#dcf5e3'),
      good:        read('--color-good',         '#0f766e'),
      warn:        read('--color-warn',         '#a1720a'),
      danger:      read('--color-danger',       '#dc2626'),
      dangerDim:   read('--color-danger-dim',   '#fbe3e2'),
    }
    // revision is the dependency that matters: it changes after the theme
    // attribute is applied, which is when computed values become correct.
  }, [theme, revision])
}

/** Compound colour resolved for canvas, theme-aware. */
export function useCompoundColour() {
  const colours = useThemeColours()
  return useCallback(
    (compound: string) => {
      switch (compound?.toUpperCase()) {
        case 'SOFT':
          return colours.soft
        case 'MEDIUM':
          return colours.medium
        case 'HARD':
          return colours.hard
        default:
          return colours.inkDim
      }
    },
    [colours],
  )
}

/**
 * Icon-only theme switch.
 *
 * Square icon button matching the design system spec. Shows the icon for the
 * theme you will *get*, not the one you are in. Inline SVG only — unicode sun/moon
 * renders as emoji on Windows, which is inconsistent with the instrument aesthetic.
 */
export function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'

  return (
    <button
      onClick={toggle}
      title={`Switch to ${next} theme`}
      aria-label={`Switch to ${next} theme`}
      className="flex h-8 w-8 items-center justify-center rounded-pill border border-line bg-card text-ink-faint shadow-card transition-colors duration-150 hover:border-line-bright hover:text-ink"
    >
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
    </button>
  )
}

function SunIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden fill="none"
      stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
      <circle cx="8" cy="8" r="2.8" />
      <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1" />
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 16 16" width="12" height="12" aria-hidden fill="none"
      stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <path d="M13.2 9.6A5.6 5.6 0 0 1 6.4 2.8a5.6 5.6 0 1 0 6.8 6.8Z" />
    </svg>
  )
}
