/**
 * The circuit, from the real racing line, with its real elevation.
 *
 * Deliberately restrained about what it colours. The model produces a state per
 * LAP, not per metre of track, so painting degradation along the tarmac would
 * imply a spatial resolution this system does not have. The track is therefore
 * neutral, and everything the model believes is carried by the car marker and
 * the panels beside it.
 *
 * Three things here were learned the hard way in Circuit3D.tsx and are repeated
 * for the same reasons:
 *   - drei's `Line`, never the `<line>` intrinsic, which collides with SVG's
 *     `line` in JSX and silently renders nothing.
 *   - The WebGL context is explicitly released on unmount. A browser allows
 *     only a handful of live contexts and force-loses the oldest without one.
 *   - A resize is dispatched after mount, because R3F's ResizeObserver can miss
 *     the first layout on a route change and leave a 300x150 canvas.
 *
 * Labels are drei `Html` rather than `<Text>`: troika's default font is fetched
 * over the network, and this has to run in a room with no internet.
 */

import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Html, Line } from '@react-three/drei'
import * as THREE from 'three'
import { useThemeColours } from '../lib/theme'
import { pointAt, ribbonGeometry, type TrackFrame } from '../lib/geometry'

/** Half the on-screen track width, in world units. Not a real track width. */
const RIBBON_HALF_WIDTH = 4.5

function TrackSurface({ frame }: { frame: TrackFrame }) {
  const colours = useThemeColours()

  const geometry = useMemo(() => {
    const { positions, indices } = ribbonGeometry(frame.points, RIBBON_HALF_WIDTH)
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    g.setIndex(new THREE.BufferAttribute(indices, 1))
    g.computeVertexNormals()
    return g
  }, [frame])

  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial color={colours.lineBright} side={THREE.DoubleSide} />
    </mesh>
  )
}

function CornerMarkers({ frame }: { frame: TrackFrame }) {
  const colours = useThemeColours()
  return (
    <group>
      {frame.corners.map(({ corner, position }) => (
        <group key={`${corner.number}${corner.letter ?? ''}`} position={position}>
          <mesh>
            <sphereGeometry args={[0.9, 10, 10]} />
            <meshBasicMaterial color={colours.lineBright} />
          </mesh>
          <Html center distanceFactor={260} zIndexRange={[10, 0]}>
            <span className="num select-none text-[11px] text-ink-faint">
              {corner.number}
              {corner.letter ?? ''}
            </span>
          </Html>
        </group>
      ))}
    </group>
  )
}

/**
 * The car, placed by real distance round the lap.
 *
 * `progress` is a fraction of the lap, not a fraction of the stint: the
 * geometry is one lap and the car repeats it.
 */
function CarMarker({
  frame,
  progress,
  colour,
}: {
  frame: TrackFrame
  progress: number
  colour: string
}) {
  const ref = useRef<THREE.Group>(null)

  useFrame(() => {
    if (!ref.current) return
    const [x, y, z] = pointAt(frame.points, progress)
    ref.current.position.set(x, y + 1.6, z)
  })

  return (
    <group ref={ref}>
      <mesh>
        <sphereGeometry args={[2.2, 16, 16]} />
        <meshBasicMaterial color={colour} />
      </mesh>
      {/* A faint halo so the car stays findable against a bright track. */}
      <mesh>
        <sphereGeometry args={[3.4, 16, 16]} />
        <meshBasicMaterial color={colour} transparent opacity={0.18} />
      </mesh>
    </group>
  )
}

function Scene({
  frame,
  progress,
  carColour,
  showCorners,
}: {
  frame: TrackFrame
  progress: number
  carColour: string
  showCorners: boolean
}) {
  const colours = useThemeColours()
  const { invalidate } = useThree()

  useEffect(() => invalidate(), [frame, invalidate])

  const linePoints = useMemo(
    () => [...frame.points, frame.points[0]].map(([x, y, z]) => new THREE.Vector3(x, y, z)),
    [frame],
  )

  /**
   * Elevation needs a reference or the eye reads height as distance.
   *
   * A flat copy of the lap sits below the track, and a comb of vertical drops
   * connects the two. Without the drops, a climb and a corner further from the
   * camera project to the same thing and Barcelona's 30 m looks like nothing.
   */
  const { groundY, shadowPoints, dropLines } = useMemo(() => {
    const ys = frame.points.map((p) => p[1])
    const lowest = Math.min(...ys)
    const spread = Math.max(...ys) - lowest
    const ground = lowest - Math.max(spread * 0.6, 8)

    const shadow = [...frame.points, frame.points[0]].map(
      ([x, , z]) => new THREE.Vector3(x, ground, z),
    )

    const every = Math.max(1, Math.round(frame.points.length / 34))
    const drops: THREE.Vector3[][] = []
    frame.points.forEach(([x, y, z], i) => {
      if (i % every !== 0) return
      drops.push([new THREE.Vector3(x, y, z), new THREE.Vector3(x, ground, z)])
    })

    return { groundY: ground, shadowPoints: shadow, dropLines: drops }
  }, [frame])

  return (
    <group>
      {dropLines.map((points, i) => (
        <Line
          key={i}
          points={points}
          color={colours.lineBright}
          lineWidth={1}
          transparent
          opacity={0.32}
        />
      ))}
      <Line points={shadowPoints} color={colours.lineBright} lineWidth={1} transparent opacity={0.85} />
      <TrackSurface frame={frame} />
      <Line points={linePoints} color={colours.lineBright} lineWidth={1.4} />
      {showCorners && <CornerMarkers frame={frame} />}

      {/* A dropped marker for start/finish too, so the loop has an anchor. */}
      <Line
        points={[
          new THREE.Vector3(frame.startFinish[0], frame.startFinish[1], frame.startFinish[2]),
          new THREE.Vector3(frame.startFinish[0], groundY, frame.startFinish[2]),
        ]}
        color={colours.good}
        lineWidth={1.2}
        transparent
        opacity={0.6}
      />

      {/* Start/finish. */}
      <mesh position={frame.startFinish}>
        <sphereGeometry args={[2.0, 14, 14]} />
        <meshBasicMaterial color={colours.good} />
      </mesh>

      <CarMarker frame={frame} progress={progress} colour={carColour} />
    </group>
  )
}

function OrbitCamera({
  frame,
  rotating,
  progress,
}: {
  frame: TrackFrame
  rotating: boolean
  progress: number
}) {
  const { camera, invalidate } = useThree()
  const angle = useRef(0.9)

  // Fit to the track rather than to a number that happened to suit Monza.
  // Zandvoort is compact and Silverstone is not, and a fixed distance either
  // crops one or strands the other in the middle of an empty frame.
  const boundingRadius = useMemo(() => {
    const far = frame.points.reduce((max, [x, , z]) => Math.max(max, Math.hypot(x, z)), 0)
    return far || 100
  }, [frame])

  useFrame((_, delta) => {
    if (rotating) angle.current += delta * 0.12
    // A raking angle rather than a plan view: elevation is the point of using
    // the real z, and from overhead a climb and a flat look identical.
    // ~40 degrees above the horizon. Shallower and the circuit reads as a flat
    // squashed line whenever the orbit swings side-on; steeper and it becomes a
    // plan view, which throws away the elevation this whole frame exists for.
    const radius = boundingRadius * 1.55
    camera.position.set(
      Math.cos(angle.current) * radius,
      boundingRadius * 1.3,
      Math.sin(angle.current) * radius,
    )
    camera.lookAt(0, 0, 0)
    invalidate()
  })

  // Keep the car moving even when the camera is parked.
  useEffect(() => invalidate(), [progress, invalidate])
  return null
}

export function RaceTrack3D({
  frame,
  progress,
  carColour,
  rotating,
  showCorners = true,
}: {
  frame: TrackFrame
  progress: number
  carColour: string
  rotating: boolean
  showCorners?: boolean
}) {
  const colours = useThemeColours()

  useEffect(() => {
    const nudge = () => window.dispatchEvent(new Event('resize'))
    const frameId = requestAnimationFrame(nudge)
    const timer = window.setTimeout(nudge, 120)
    return () => {
      cancelAnimationFrame(frameId)
      clearTimeout(timer)
    }
  }, [])

  const glRef = useRef<THREE.WebGLRenderer | null>(null)
  useEffect(
    () => () => {
      const gl = glRef.current
      if (!gl) return
      gl.forceContextLoss()
      gl.dispose()
      glRef.current = null
    },
    [],
  )

  return (
    <div className="h-full w-full" style={{ background: colours.ground }}>
      <Canvas
        frameloop="always"
        camera={{ position: [150, 120, 150], fov: 46, near: 1, far: 4000 }}
        dpr={[1, 2]}
        gl={{ antialias: true }}
        resize={{ debounce: 0, scroll: false }}
        onCreated={({ gl }) => {
          glRef.current = gl
        }}
      >
        <OrbitCamera frame={frame} rotating={rotating} progress={progress} />
        <Scene
          frame={frame}
          progress={progress}
          carColour={carColour}
          showCorners={showCorners}
        />
      </Canvas>
    </div>
  )
}
