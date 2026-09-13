/**
 * Real circuit geometry, turned into something three.js can draw.
 *
 * The JSON under `circuits/` is a byte-for-byte copy of `data/geometry/*.json`,
 * written by `scripts/export_track_geometry.py`. It is the racing line of the
 * session's fastest lap, resampled to 600 points evenly spaced by distance,
 * with `z` taken from F1's positioning feed rather than synthesised. Nothing in
 * this file invents a coordinate.
 *
 * It lives in the bundle rather than behind an endpoint on purpose. The only
 * geometry route the API exposes (`/api/physics/track-geometry`) serves a
 * different, telemetry-derived file under `data/demo/`, and it has no Sakhir or
 * Spielberg. Bundling keeps the centrepiece view working with no network and no
 * backend route, which is the condition it will actually be demonstrated under.
 *
 * Three transforms happen here and each one is a decision worth naming:
 *
 *   1. **Rotation.** `rotation_deg` is what broadcast applies. Skip it and the
 *      map sits sideways, which every viewer who knows the circuit notices
 *      immediately.
 *   2. **Normalisation.** Every circuit is centred and scaled so the longer of
 *      its two ground axes spans `WORLD_EXTENT`. One camera then frames Sakhir
 *      and Spielberg identically.
 *   3. **Elevation exaggeration.** Held apart from the ground scale and stated
 *      on screen. Spielberg climbs 63 m over 4.3 km; at true proportions that
 *      hill is a millimetre and the render lies by understatement. The factor
 *      is reported so a viewer knows the vertical is not to scale.
 *
 * Curvature is computed here too, from the geometry alone. Because the points
 * are evenly spaced by arc length, the second difference is proportional to the
 * curvature, so no vehicle model and no telemetry is needed -- the same
 * reasoning the existing circuit view uses for its cornering-load colouring.
 */

import * as THREE from 'three'

export interface CircuitCorner {
  number: number
  letter: string | null
  x: number
  y: number
  angle_deg: number
  distance_m: number
}

export interface CircuitGeometry {
  session_id?: string
  circuit: string
  event?: string
  year?: number
  units: string
  rotation_deg: number
  lap_length_m: number
  elevation_range_m: [number, number]
  elevation_gain_m: number
  n_points: number
  centreline: [number, number, number][]
  corners: CircuitCorner[]
  start_finish: [number, number, number]
}

/** Longer ground axis of any circuit, in world units. One camera fits them all. */
const WORLD_EXTENT = 100

/**
 * Vertical multiplier, applied on top of the ground scale.
 *
 * Stated on screen wherever the hills are visible. See the module note: a
 * truthful vertical is an invisible one, so the choice is to exaggerate and say
 * so rather than to flatten and say nothing.
 */
export const ELEVATION_EXAGGERATION = 7

/**
 * Half-width of the drawn ribbon, world units.
 *
 * Not to scale. A real 12 m track at this ground scale is a thread about a
 * fifth of a unit wide -- invisible on a projector at three metres. The ribbon
 * is drawn wide enough to read and is a stylised road, not a measurement of
 * track limits. The line down its middle is the part that is real.
 */
const TRACK_HALF_WIDTH = 1.25

export interface TrackModel {
  /** Circuit metadata, straight from the file. */
  meta: CircuitGeometry
  /** Racing line in world space, closed, 600 points. */
  points: THREE.Vector3[]
  /** Cumulative arc length in world units at each point, plus the total. */
  arc: number[]
  totalArc: number
  /** Curvature at each point, normalised to 0..1 over this circuit. */
  load: number[]
  /** Corner markers, positioned on the line by their recorded lap distance. */
  corners: { corner: CircuitCorner; position: THREE.Vector3; fraction: number }[]
  /** Start/finish, and the fraction of a lap at which it sits (0). */
  startFinish: THREE.Vector3
  /** Metres of real circuit per world unit -- for anything that must report metres. */
  metresPerUnit: number
}

function rotate(x: number, y: number, degrees: number): [number, number] {
  const t = (degrees * Math.PI) / 180
  const c = Math.cos(t)
  const s = Math.sin(t)
  return [x * c - y * s, x * s + y * c]
}

/**
 * Build the drawable model for one circuit.
 *
 * The loop is closed in the source file (last point meets the first), so no
 * point is appended here; doing so would put two coincident vertices at the
 * start line and produce a zero-length tangent exactly where the car spawns.
 */
export function buildTrackModel(meta: CircuitGeometry): TrackModel {
  const rotated = meta.centreline.map(([x, y, z]) => {
    const [rx, ry] = rotate(x, y, meta.rotation_deg)
    return [rx, ry, z] as [number, number, number]
  })

  const xs = rotated.map((p) => p[0])
  const ys = rotated.map((p) => p[1])
  const zs = rotated.map((p) => p[2])
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2
  const cy = (Math.max(...ys) + Math.min(...ys)) / 2
  const cz = Math.min(...zs)
  const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
  const scale = WORLD_EXTENT / (extent || 1)

  // y is up in three.js; the circuit's y becomes the world's z.
  const points = rotated.map(
    ([x, y, z]) =>
      new THREE.Vector3(
        (x - cx) * scale,
        (z - cz) * scale * ELEVATION_EXAGGERATION,
        (y - cy) * scale,
      ),
  )

  const arc: number[] = [0]
  for (let i = 1; i < points.length; i += 1) {
    arc.push(arc[i - 1] + points[i].distanceTo(points[i - 1]))
  }
  const totalArc = arc[arc.length - 1] + points[0].distanceTo(points[points.length - 1])

  // Curvature from the second difference. The points are evenly spaced by
  // distance, which is what makes this valid without a chord-length correction.
  const raw = points.map((_, i) => {
    const prev = points[(i - 1 + points.length) % points.length]
    const next = points[(i + 1) % points.length]
    const here = points[i]
    return new THREE.Vector3(
      prev.x - 2 * here.x + next.x,
      0, // curvature of the ground path; the hill is not cornering load
      prev.z - 2 * here.z + next.z,
    ).length()
  })
  // Smooth over ~5 points: a single-sample second difference is noisy enough
  // that straights flicker, and the eye reads flicker as data.
  const smoothed = raw.map((_, i) => {
    let total = 0
    for (let k = -2; k <= 2; k += 1) total += raw[(i + k + raw.length) % raw.length]
    return total / 5
  })
  const hi = Math.max(...smoothed) || 1
  const load = smoothed.map((v) => Math.min(1, v / hi))

  const metresPerUnit = meta.lap_length_m / totalArc

  const corners = meta.corners.map((corner) => {
    const fraction = Math.min(0.999, Math.max(0, corner.distance_m / meta.lap_length_m))
    return { corner, position: pointAtFraction(points, arc, totalArc, fraction), fraction }
  })

  return {
    meta,
    points,
    arc,
    totalArc,
    load,
    corners,
    startFinish: points[0].clone(),
    metresPerUnit,
  }
}

/** Interpolated position at a fraction of a lap, 0 at the start line. */
export function pointAtFraction(
  points: THREE.Vector3[],
  arc: number[],
  totalArc: number,
  fraction: number,
): THREE.Vector3 {
  const target = ((fraction % 1) + 1) % 1 * totalArc
  let i = 0
  while (i < arc.length - 1 && arc[i + 1] <= target) i += 1
  const a = points[i]
  const b = points[(i + 1) % points.length]
  const segment = (i + 1 < arc.length ? arc[i + 1] : totalArc) - arc[i]
  const t = segment > 0 ? (target - arc[i]) / segment : 0
  return a.clone().lerp(b, t)
}

/** Index of the sample nearest a lap fraction. Used for per-point lookups. */
export function indexAtFraction(track: TrackModel, fraction: number): number {
  const target = ((fraction % 1) + 1) % 1 * track.totalArc
  let i = 0
  while (i < track.arc.length - 1 && track.arc[i + 1] <= target) i += 1
  return i
}

/** Unit tangent at a lap fraction, for pointing the car down the road. */
export function tangentAtFraction(track: TrackModel, fraction: number): THREE.Vector3 {
  const i = indexAtFraction(track, fraction)
  const a = track.points[i]
  const b = track.points[(i + 1) % track.points.length]
  return b.clone().sub(a).normalize()
}

/**
 * A ribbon along the racing line between two lap fractions.
 *
 * Returns positions and the sample index each vertex came from, so a caller can
 * colour by anything it has per sample -- curvature here, and the same vertices
 * are reused for the pit-window glow.
 */
export function ribbonGeometry(
  track: TrackModel,
  from: number,
  to: number,
  halfWidth = TRACK_HALF_WIDTH,
  lift = 0,
): { geometry: THREE.BufferGeometry; sampleIndex: number[] } {
  const span = to - from
  const steps = Math.max(2, Math.round(Math.abs(span) * track.points.length))
  const positions = new Float32Array((steps + 1) * 2 * 3)
  const sampleIndex: number[] = []

  const up = new THREE.Vector3(0, 1, 0)
  for (let s = 0; s <= steps; s += 1) {
    const fraction = from + (span * s) / steps
    const centre = pointAtFraction(track.points, track.arc, track.totalArc, fraction)
    const tangent = tangentAtFraction(track, fraction)
    const side = new THREE.Vector3().crossVectors(tangent, up).normalize().multiplyScalar(halfWidth)
    const base = s * 6
    positions[base + 0] = centre.x - side.x
    positions[base + 1] = centre.y + lift
    positions[base + 2] = centre.z - side.z
    positions[base + 3] = centre.x + side.x
    positions[base + 4] = centre.y + lift
    positions[base + 5] = centre.z + side.z
    sampleIndex.push(indexAtFraction(track, fraction))
  }

  const indices: number[] = []
  for (let s = 0; s < steps; s += 1) {
    const a = s * 2
    indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setIndex(indices)
  geometry.computeVertexNormals()
  return { geometry, sampleIndex }
}
