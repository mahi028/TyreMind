/**
 * The bundled circuit files.
 *
 * Imported with Vite's `?raw` so the JSON reaches the bundle byte-for-byte as
 * it sits in `data/geometry/`, rather than being rewritten into a TypeScript
 * literal where a digit could quietly change. It also sidesteps needing
 * `resolveJsonModule` in a tsconfig this view does not own.
 *
 * Two circuits, because two cases are demonstrated: Sakhir for the 2025 Bahrain
 * Grand Prix, Spielberg for Austria. Adding another is a copy of the file from
 * `data/geometry/` and one line here.
 */

import sakhirRaw from './circuits/sakhir.json?raw'
import spielbergRaw from './circuits/spielberg.json?raw'
import type { CircuitGeometry } from './geometry'

export const CIRCUITS: Record<string, CircuitGeometry> = {
  sakhir: JSON.parse(sakhirRaw) as CircuitGeometry,
  spielberg: JSON.parse(spielbergRaw) as CircuitGeometry,
}
