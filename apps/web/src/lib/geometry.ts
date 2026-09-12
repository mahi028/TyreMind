/**
 * Real circuit geometry, as exported by scripts/export_track_geometry.py.
 *
 * The files under public/geometry/ are the fastest lap's actual racing line,
 * resampled evenly by distance, in metres, with z taken from F1's positioning
 * feed. They are shipped with the frontend rather than fetched from the API for
 * one reason: the exporter needs the network, and the demo has to survive a
 * room with no internet.
 *
 * Nothing here invents a number. The only liberty taken is vertical
 * exaggeration, which is returned alongside the geometry so the view can say
 * so on screen instead of quietly overselling a hill.
 */

export interface TrackCorner {
  number: number
  letter: string | null
  x: number
  y: number
  angle_deg: number
  distance_m: number
}

export interface TrackGeometryFile {
  session_id: string
  circuit: string
  event: string
  year: number
  units: 'metres'
  rotation_deg: number
  lap_length_m: number
  elevation_range_m: [number, number]
  elevation_gain_m: number
  n_points: number
  /** [x, y, z] in metres. z is real elevation. */
  centreline: [number, number, number][]
  corners: TrackCorner[]
  start_finish: [number, number, number]
}

/** Circuits with an exported geometry file, keyed by the API's grand_prix name. */
export const GEOMETRY_SLUG: Record<string, string> = {
  Barcelona: 'barcelona',
  Monza: 'monza',
  Silverstone: 'silverstone',
  Zandvoort: 'zandvoort',
}

export function geometrySlug(grandPrix: string): string | null {
  return GEOMETRY_SLUG[grandPrix] ?? null
}

export async function loadTrackGeometry(slug: string): Promise<TrackGeometryFile> {
  const response = await fetch(`${import.meta.env.BASE_URL}geometry/${slug}.json`)
  if (!response.ok) {
    throw new Error(
      `No geometry for "${slug}". Run: python scripts/export_track_geometry.py --all-demo`,
    )
  }
  return response.json() as Promise<TrackGeometryFile>
}

// ---------------------------------------------------------------------------
// Metres to world units
// ---------------------------------------------------------------------------

/** Longest horizontal dimension of the track, in world units, after scaling. */
const WORLD_EXTENT = 200

/**
 * Elevation is multiplied by this so it is visible at all.
 *
 * Barcelona climbs 29.9 m over a 4,601 m lap. At true proportions that is 0.6%
 * of the track's own width on screen, which renders as a flat plate. The view
 * prints the factor next to the elevation figure so nobody reads the hills as
 * being to scale.
 */
export const ELEVATION_EXAGGERATION = 12

export interface TrackFrame {
  /** Centreline in world units, Y-up, rotated, centred and scaled. */
  points: [number, number, number][]
  /** Corner markers in the same world frame, sitting on the surface. */
  corners: { corner: TrackCorner; position: [number, number, number] }[]
  startFinish: [number, number, number]
  /** World units per metre, for anything that needs to talk in real distance. */
  unitsPerMetre: number
  elevationExaggeration: number
}

/**
 * Put the track in a frame three.js can render.
 *
 * Applies `rotation_deg` (which orients the map the way broadcast does),
 * centres on the bounding box, scales the longest axis to a fixed extent, and
 * maps [x, y, z] metres onto three.js's Y-up convention as (x, z, y).
 */
export function toWorldFrame(geometry: TrackGeometryFile): TrackFrame {
  const radians = (geometry.rotation_deg * Math.PI) / 180
  const cos = Math.cos(radians)
  const sin = Math.sin(radians)
  const rotate = (x: number, y: number): [number, number] => [
    x * cos - y * sin,
    x * sin + y * cos,
  ]

  const rotated = geometry.centreline.map(([x, y, z]) => {
    const [rx, ry] = rotate(x, y)
    return [rx, ry, z] as [number, number, number]
  })

  const xs = rotated.map((p) => p[0])
  const ys = rotated.map((p) => p[1])
  const zs = rotated.map((p) => p[2])
  const centreX = (Math.max(...xs) + Math.min(...xs)) / 2
  const centreY = (Math.max(...ys) + Math.min(...ys)) / 2
  const centreZ = (Math.max(...zs) + Math.min(...zs)) / 2

  const extent = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
  const unitsPerMetre = WORLD_EXTENT / (extent || 1)

  const place = (x: number, y: number, z: number): [number, number, number] => [
    (x - centreX) * unitsPerMetre,
    (z - centreZ) * unitsPerMetre * ELEVATION_EXAGGERATION,
    (y - centreY) * unitsPerMetre,
  ]

  const points = rotated.map(([x, y, z]) => place(x, y, z))

  // Corners carry no elevation of their own, so each one borrows the height of
  // the nearest centreline point rather than floating at zero.
  const corners = geometry.corners.map((corner) => {
    const [rx, ry] = rotate(corner.x, corner.y)
    let nearest = 0
    let best = Infinity
    rotated.forEach(([x, y], i) => {
      const d = (x - rx) ** 2 + (y - ry) ** 2
      if (d < best) {
        best = d
        nearest = i
      }
    })
    return { corner, position: place(rx, ry, rotated[nearest][2]) }
  })

  const [sfx, sfy, sfz] = geometry.start_finish
  const [rsfx, rsfy] = rotate(sfx, sfy)

  return {
    points,
    corners,
    startFinish: place(rsfx, rsfy, sfz),
    unitsPerMetre,
    elevationExaggeration: ELEVATION_EXAGGERATION,
  }
}

/**
 * Build a track surface from the centreline.
 *
 * Two vertices per centreline point, offset either side of the direction of
 * travel, stitched into triangles and closed back to the start. The racing line
 * is not the kerb, so the ribbon is only an honest indication of where the track
 * is; it is deliberately narrow rather than pretending to be track edges.
 */
export function ribbonGeometry(
  points: [number, number, number][],
  halfWidth: number,
): { positions: Float32Array; indices: Uint32Array } {
  const n = points.length
  const positions = new Float32Array(n * 2 * 3)
  const indices = new Uint32Array(n * 6)

  for (let i = 0; i < n; i++) {
    const prev = points[(i - 1 + n) % n]
    const next = points[(i + 1) % n]
    // Tangent from the neighbours, flattened: the ribbon should lie across the
    // direction of travel, not tilt with the gradient.
    const tx = next[0] - prev[0]
    const tz = next[2] - prev[2]
    const length = Math.hypot(tx, tz) || 1
    // Left-hand normal in the ground plane.
    const nx = -tz / length
    const nz = tx / length

    const [x, y, z] = points[i]
    positions[i * 6 + 0] = x + nx * halfWidth
    positions[i * 6 + 1] = y
    positions[i * 6 + 2] = z + nz * halfWidth
    positions[i * 6 + 3] = x - nx * halfWidth
    positions[i * 6 + 4] = y
    positions[i * 6 + 5] = z - nz * halfWidth

    const a = i * 2
    const b = i * 2 + 1
    const c = ((i + 1) % n) * 2
    const d = ((i + 1) % n) * 2 + 1
    indices[i * 6 + 0] = a
    indices[i * 6 + 1] = b
    indices[i * 6 + 2] = c
    indices[i * 6 + 3] = b
    indices[i * 6 + 4] = d
    indices[i * 6 + 5] = c
  }

  return { positions, indices }
}

/** Position along the closed loop at `t` in [0, 1), interpolated between points. */
export function pointAt(
  points: [number, number, number][],
  t: number,
): [number, number, number] {
  const n = points.length
  const exact = ((t % 1) + 1) % 1 * n
  const i = Math.floor(exact)
  const frac = exact - i
  const a = points[i % n]
  const b = points[(i + 1) % n]
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ]
}
