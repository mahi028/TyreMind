/**
 * The circuit, the car, and the belief drawn onto the road.
 *
 * Four things are rendered and they answer to different authorities:
 *
 *   - **The road.** Real geometry, real elevation, unmodified. Its *width* is
 *     stylised — a 12 m track at this scale is a thread — and the thin bright
 *     line down the middle is where the real coordinates are.
 *   - **The heat along the road.** Curvature computed from the geometry, scaled
 *     by the model's current degradation rate. So the bright places are the
 *     corners that work the tyre, and the whole lap brightens as the set wears.
 *     Curvature is measured; the brightness mapping is a display choice.
 *   - **The window on the track.** The band across the start/finish straight
 *     lights when the current lap falls inside the optimiser's one-second
 *     window. Its two edges are the window's two laps, not decoration.
 *   - **The curtain ahead of the car.** Height is the projected tyre loss over
 *     the road still to come; the translucent skirt is the 95% interval on it.
 *     It is wider than it is tall, which is the honest shape of a projection
 *     and the reason it is drawn rather than printed.
 *
 * No asset is fetched. The car is built from primitives, the fonts are the
 * page's own, and nothing here touches the network — the venue's wifi is not a
 * dependency of the centrepiece.
 */

import { useEffect, useMemo, useRef, type RefObject } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Line, OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import {
  buildTrackModel,
  pointAtFraction,
  ribbonGeometry,
  tangentAtFraction,
  type CircuitGeometry,
  type TrackModel,
} from './geometry'
import { useThemeColours } from '../../lib/theme'

export type CameraMode = 'chase' | 'overview'

export interface SceneState {
  /** Position around the lap, 0 at the start line. */
  lapFraction: number
  /** Degradation heat, 0..1, for the road colouring. */
  heat: number
  /** Projected loss over the rest of the lap, seconds, and its 95% half-width. */
  projectedLoss: number
  projectedHalfWidth: number
  /** True while the window is open on this lap. */
  windowOpen: boolean
  /** True on the lap the driver actually boxed. */
  boxingNow: boolean
  compoundColour: string
}

function useDisposable<T extends { dispose(): void }>(value: T): T {
  useEffect(() => () => value.dispose(), [value])
  return value
}

/** Three-stop ramp: cool where the tyre is idle, hot where it is working. */
function ramp(t: number, cold: THREE.Color, warm: THREE.Color, hot: THREE.Color): THREE.Color {
  const c = new THREE.Color()
  if (t < 0.5) c.lerpColors(cold, warm, t * 2)
  else c.lerpColors(warm, hot, (t - 0.5) * 2)
  return c
}

function Road({ track, heat }: { track: TrackModel; heat: number }) {
  const colours = useThemeColours()
  const { geometry, sampleIndex } = useMemo(
    () => ribbonGeometry(track, 0, 1, 1.25, 0.02),
    [track],
  )
  useDisposable(geometry)

  useEffect(() => {
    const cold = new THREE.Color(colours.fuel).multiplyScalar(0.45)
    const warm = new THREE.Color(colours.medium)
    const hot = new THREE.Color(colours.alert)
    const count = sampleIndex.length * 2
    const array = new Float32Array(count * 3)
    sampleIndex.forEach((index, s) => {
      // Curvature sets where the tyre is worked; the live rate sets how hard.
      const value = Math.min(1, 0.12 * heat + track.load[index] * (0.35 + 0.75 * heat))
      const colour = ramp(value, cold, warm, hot)
      for (const side of [0, 1]) {
        const base = (s * 2 + side) * 3
        array[base] = colour.r
        array[base + 1] = colour.g
        array[base + 2] = colour.b
      }
    })
    geometry.setAttribute('color', new THREE.BufferAttribute(array, 3))
    geometry.attributes.color.needsUpdate = true
  }, [geometry, sampleIndex, track, heat, colours])

  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial
        vertexColors
        side={THREE.DoubleSide}
        roughness={0.72}
        metalness={0.05}
        emissiveIntensity={0.25}
      />
    </mesh>
  )
}

/**
 * Elevation, drawn as a comb of droppers to the base plane.
 *
 * Without something under it the road is a ribbon floating at varying heights,
 * which reads as a flat road seen at an angle — the hill disappears. A solid
 * skirt does the job and dominates the frame; a sparse comb gives the eye the
 * same vertical reference for a fraction of the ink. Spielberg climbs 63 m and
 * this is what makes that visible.
 */
function Droppers({ track }: { track: TrackModel }) {
  const colours = useThemeColours()
  const EVERY = 8
  const geometry = useMemo(() => {
    const base = Math.min(...track.points.map((p) => p.y)) - 1
    const positions: number[] = []
    track.points.forEach((p, i) => {
      if (i % EVERY !== 0) return
      positions.push(p.x, p.y, p.z, p.x, base, p.z)
    })
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    return g
  }, [track])
  useDisposable(geometry)

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color={colours.lineBright} transparent opacity={0.3} />
    </lineSegments>
  )
}

/** Corner posts, placed by the recorded lap distance of each corner. */
function Corners({ track }: { track: TrackModel }) {
  const colours = useThemeColours()
  return (
    <group>
      {track.corners.map(({ corner, position }) => (
        <mesh key={`${corner.number}${corner.letter ?? ''}`} position={[position.x, position.y + 0.9, position.z]}>
          <cylinderGeometry args={[0.09, 0.09, 1.8, 6]} />
          <meshBasicMaterial color={colours.inkFaint} transparent opacity={0.5} />
        </mesh>
      ))}
    </group>
  )
}

/** Start/finish gantry, on the real start/finish coordinate. */
function StartFinish({ track }: { track: TrackModel }) {
  const colours = useThemeColours()
  const { geometry } = useMemo(() => ribbonGeometry(track, -0.004, 0.004, 1.3, 0.06), [track])
  useDisposable(geometry)
  const tangent = useMemo(() => tangentAtFraction(track, 0), [track])
  const side = useMemo(
    () => new THREE.Vector3().crossVectors(tangent, new THREE.Vector3(0, 1, 0)).normalize(),
    [tangent],
  )
  const p = track.startFinish

  return (
    <group>
      <mesh geometry={geometry}>
        <meshBasicMaterial color={colours.ink} transparent opacity={0.75} side={THREE.DoubleSide} />
      </mesh>
      {[-1, 1].map((sign) => (
        <mesh
          key={sign}
          position={[p.x + side.x * 1.7 * sign, p.y + 1.6, p.z + side.z * 1.7 * sign]}
        >
          <boxGeometry args={[0.16, 3.2, 0.16]} />
          <meshStandardMaterial color={colours.inkFaint} roughness={0.6} />
        </mesh>
      ))}
      <mesh position={[p.x, p.y + 3.2, p.z]}>
        <boxGeometry args={[3.6, 0.18, 0.18]} />
        <meshStandardMaterial color={colours.inkDim} roughness={0.6} />
      </mesh>
    </group>
  )
}

/**
 * The pit window, painted on the road where the pit entry is.
 *
 * Placed over the last twelfth of the lap because that is where a car commits
 * to the lane. It is lit only while the optimiser's window contains the lap the
 * car is on, so a viewer sees the decision arrive rather than being told it.
 */
function PitWindow({ track, open, boxing }: { track: TrackModel; open: boolean; boxing: boolean }) {
  const colours = useThemeColours()
  const { geometry } = useMemo(() => ribbonGeometry(track, 0.915, 0.998, 1.28, 0.05), [track])
  useDisposable(geometry)
  const material = useRef<THREE.MeshBasicMaterial>(null)

  useFrame((state) => {
    if (!material.current) return
    const pulse = 0.5 + 0.5 * Math.sin(state.clock.elapsedTime * (boxing ? 7 : 2.4))
    material.current.opacity = open || boxing ? 0.22 + 0.5 * pulse : 0.0
    material.current.color.set(boxing ? colours.good : colours.alert)
  })

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial
        ref={material}
        transparent
        opacity={0}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  )
}

/**
 * The projection ahead of the car, drawn as a curtain over the road still to run.
 *
 * Height is the model's expected tyre loss across that stretch. The pale skirt
 * around it is the 95% interval, and it is deliberately not clamped: at a
 * ten-lap horizon the band is wider than the line, which is the true state of
 * the knowledge and the single most important thing this product has to say.
 */
function ProjectionCurtain({
  track,
  state,
}: {
  track: TrackModel
  state: RefObject<SceneState>
}) {
  const colours = useThemeColours()
  const mean = useRef<THREE.Mesh>(null)
  const band = useRef<THREE.Mesh>(null)
  const SEGMENTS = 90

  const geometries = useMemo(() => {
    const make = () => {
      const positions = new Float32Array((SEGMENTS + 1) * 2 * 3)
      const indices: number[] = []
      for (let s = 0; s < SEGMENTS; s += 1) {
        const a = s * 2
        indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2)
      }
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
      g.setIndex(indices)
      return g
    }
    return { meanGeometry: make(), bandGeometry: make() }
  }, [])
  useDisposable(geometries.meanGeometry)
  useDisposable(geometries.bandGeometry)

  useFrame(() => {
    const s = state.current
    if (!s) return
    const write = (geometry: THREE.BufferGeometry, height: number, floor: number) => {
      const array = geometry.attributes.position.array as Float32Array
      for (let i = 0; i <= SEGMENTS; i += 1) {
        // Only the road still ahead on this lap. Wrapping a horizon of ten laps
        // around the circuit ten times would be a scribble, not a projection;
        // the full horizon is the chart in the panel below.
        const t = i / SEGMENTS
        const fraction = s.lapFraction + t * (1 - s.lapFraction) * 0.25
        const p = pointAtFraction(track.points, track.arc, track.totalArc, fraction)
        const grow = t * t // loss accumulates faster than linearly with distance
        array[i * 6] = p.x
        array[i * 6 + 1] = p.y + 0.05 + floor * grow
        array[i * 6 + 2] = p.z
        array[i * 6 + 3] = p.x
        array[i * 6 + 4] = p.y + 0.05 + height * grow
        array[i * 6 + 5] = p.z
      }
      geometry.attributes.position.needsUpdate = true
      geometry.computeBoundingSphere()
    }
    // One second of projected loss is drawn as two world units of height.
    const SCALE = 1.3
    write(geometries.meanGeometry, s.projectedLoss * SCALE, 0)
    write(
      geometries.bandGeometry,
      (s.projectedLoss + s.projectedHalfWidth) * SCALE,
      Math.max(0, s.projectedLoss - s.projectedHalfWidth) * SCALE,
    )
    if (mean.current) mean.current.visible = s.projectedLoss > 0.01
    if (band.current) band.current.visible = s.projectedLoss > 0.01
  })

  return (
    <group>
      <mesh ref={band} geometry={geometries.bandGeometry}>
        <meshBasicMaterial
          color={colours.alert}
          transparent
          opacity={0.12}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <mesh ref={mean} geometry={geometries.meanGeometry}>
        <meshBasicMaterial
          color={colours.alert}
          transparent
          opacity={0.34}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
    </group>
  )
}

/**
 * The car. Built from primitives, on purpose.
 *
 * A downloaded model would be a network dependency and a licence question for
 * about four seconds of extra realism. The silhouette that matters at
 * projector distance is low body, exposed wheels, two wings — that is what is
 * here, and the wheel colour is the live compound, which is the one thing on
 * the car carrying data.
 */
function Car({
  track,
  state,
  cameraMode,
}: {
  track: TrackModel
  state: RefObject<SceneState>
  cameraMode: CameraMode
}) {
  const colours = useThemeColours()
  const group = useRef<THREE.Group>(null)
  const wheels = useRef<THREE.MeshStandardMaterial[]>([])
  const { camera } = useThree()
  const look = useMemo(() => new THREE.Vector3(), [])
  const desired = useMemo(() => new THREE.Vector3(), [])

  useFrame((_, delta) => {
    const s = state.current
    if (!s || !group.current) return
    const position = pointAtFraction(track.points, track.arc, track.totalArc, s.lapFraction)
    const tangent = tangentAtFraction(track, s.lapFraction)
    group.current.position.set(position.x, position.y + 0.28, position.z)
    look.copy(position).add(tangent)
    group.current.lookAt(look.x, position.y + 0.28, look.z)

    for (const material of wheels.current) {
      if (material) material.color.set(s.compoundColour)
    }

    if (cameraMode === 'chase') {
      // Behind and above, easing in. The easing is frame-rate independent so a
      // slow projector and a fast laptop show the same motion.
      desired
        .copy(position)
        .addScaledVector(tangent, -12)
        .add(new THREE.Vector3(0, 5.5, 0))
      // Ease normally, but snap when the camera is nowhere near where it should
      // be -- entering chase from the overview, or a jump to another lap. Easing
      // across half a circuit takes seconds the viewer reads as a broken camera.
      if (camera.position.distanceTo(desired) > 40) camera.position.copy(desired)
      else camera.position.lerp(desired, 1 - Math.exp(-3.2 * delta))
      camera.lookAt(position.x, position.y + 1.4, position.z)
    }
  })

  const body = colours.theme === 'dark' ? '#cfd8dd' : '#2a3942'

  return (
    <group ref={group}>
      <mesh position={[0, 0.16, 0.1]} castShadow>
        <boxGeometry args={[0.52, 0.22, 2.3]} />
        <meshStandardMaterial color={body} roughness={0.35} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.3, -0.25]}>
        <boxGeometry args={[0.42, 0.3, 0.6]} />
        <meshStandardMaterial color={body} roughness={0.35} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.1, 1.42]}>
        <boxGeometry args={[1.15, 0.06, 0.34]} />
        <meshStandardMaterial color={colours.alert} roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.46, -1.15]}>
        <boxGeometry args={[1.0, 0.28, 0.1]} />
        <meshStandardMaterial color={colours.alert} roughness={0.4} />
      </mesh>
      {[
        [0.62, 0.9],
        [-0.62, 0.9],
        [0.66, -0.85],
        [-0.66, -0.85],
      ].map(([x, z], i) => (
        <mesh key={i} position={[x, 0.26, z]} rotation={[0, 0, Math.PI / 2]}>
          <cylinderGeometry args={[0.26, 0.26, 0.24, 14]} />
          <meshStandardMaterial
            ref={(m) => {
              if (m) wheels.current[i] = m
            }}
            color={colours.soft}
            roughness={0.85}
          />
        </mesh>
      ))}
    </group>
  )
}

function Rig({ cameraMode }: { cameraMode: CameraMode }) {
  const { camera } = useThree()
  useEffect(() => {
    if (cameraMode === 'overview') {
      camera.position.set(0, 96, 112)
      camera.lookAt(0, 0, 0)
    }
  }, [cameraMode, camera])
  return null
}

export function TrackScene({
  geometry,
  state,
  heat,
  windowOpen,
  boxingNow,
  cameraMode,
}: {
  geometry: CircuitGeometry
  state: RefObject<SceneState>
  /** Changes once per lap, so the road recolours on a lap boundary not a frame. */
  heat: number
  windowOpen: boolean
  boxingNow: boolean
  cameraMode: CameraMode
}) {
  const colours = useThemeColours()
  const track = useMemo(() => buildTrackModel(geometry), [geometry])
  const linePoints = useMemo(
    () => [...track.points, track.points[0]].map((p) => [p.x, p.y + 0.06, p.z] as [number, number, number]),
    [track],
  )

  return (
    <Canvas
      camera={{ position: [0, 96, 112], fov: 42, near: 0.3, far: 2000 }}
      dpr={[1, 2]}
      gl={{ antialias: true }}
      style={{ background: colours.ground }}
    >
      <color attach="background" args={[colours.ground]} />
      <fog attach="fog" args={[colours.ground, 160, 460]} />

      <ambientLight intensity={colours.theme === 'dark' ? 0.55 : 1.0} />
      <directionalLight position={[60, 90, 40]} intensity={colours.theme === 'dark' ? 1.5 : 2.0} />
      <directionalLight position={[-70, 30, -50]} intensity={0.5} color={colours.fuel} />

      <Rig cameraMode={cameraMode} />
      <Droppers track={track} />
      <Road track={track} heat={heat} />
      <Line points={linePoints} color={colours.ink} lineWidth={1.1} transparent opacity={0.35} />
      <Corners track={track} />
      <StartFinish track={track} />
      <PitWindow track={track} open={windowOpen} boxing={boxingNow} />
      <ProjectionCurtain track={track} state={state} />
      <Car track={track} state={state} cameraMode={cameraMode} />

      {cameraMode === 'overview' && (
        <OrbitControls enablePan makeDefault target={[0, 0, 0]} maxPolarAngle={Math.PI / 2.05} />
      )}
    </Canvas>
  )
}
