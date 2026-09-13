/**
 * The tyre digital twin: a car seen from above, with each corner drawn as what
 * the model believes about it.
 *
 * This is the one place the physics layer becomes something you can look at.
 * Curvature recovered from GPS traces gives lateral acceleration, load transfer
 * splits that across the four corners, and the result is that a clockwise
 * circuit visibly works its left-hand tyres harder. The asymmetry on screen is
 * measured, not decorative -- `exp06_circuit_asymmetry` recovers circuit
 * rotation direction from it on 7 of 8 circuits.
 *
 * Drawn as inline SVG rather than a chart library because the geometry *is* the
 * information: where a tyre sits on the car is what the reader needs to map the
 * number onto the real object.
 */

import { compoundColour } from '../lib/api'

export interface CornerLoad {
  FL: number
  FR: number
  RL: number
  RR: number
}

function healthColour(health: number): string {
  if (health > 60) return 'var(--color-good)'
  if (health > 25) return 'var(--color-medium)'
  return 'var(--color-danger)'
}

/**
 * The car, drawn from above, centred on x = 140 in a 280 x 300 viewBox.
 *
 * Laid out from the real thing rather than eyeballed: front wing, nose, halo,
 * sidepods with their intakes, engine cover, floor, rear wing. The rear tyres
 * are wider than the fronts because they are (405 mm against 305 mm), and a
 * reader who knows the car should not find anything here that contradicts it.
 *
 * The margins either side are deliberate. The four readouts live out there, so
 * each number sits next to the corner it describes.
 */
const CAR = {
  centreX: 140,
  frontAxleY: 78,
  rearAxleY: 245,
  // Front tyres are 305 mm, rears 405 mm, both about 720 mm across. At the
  // scale below that is 14, 19 and 33 units, and the wheelbase is 167.
  frontTyre: { width: 14, height: 33, offset: 37 },
  rearTyre: { width: 19, height: 33, offset: 37 },
}

const CORNERS: {
  key: keyof CornerLoad
  label: string
  /** Where the tyre sits. */
  x: number
  y: number
  /** Where the readout sits, and which way it reads. */
  labelX: number
  labelY: number
  anchor: 'start' | 'end'
}[] = [
  {
    key: 'FL',
    label: 'front left',
    x: CAR.centreX - CAR.frontTyre.offset,
    y: CAR.frontAxleY,
    labelX: 8,
    labelY: 62,
    anchor: 'start',
  },
  {
    key: 'FR',
    label: 'front right',
    x: CAR.centreX + CAR.frontTyre.offset,
    y: CAR.frontAxleY,
    labelX: 272,
    labelY: 62,
    anchor: 'end',
  },
  {
    key: 'RL',
    label: 'rear left',
    x: CAR.centreX - CAR.rearTyre.offset,
    y: CAR.rearAxleY,
    labelX: 8,
    labelY: 262,
    anchor: 'start',
  },
  {
    key: 'RR',
    label: 'rear right',
    x: CAR.centreX + CAR.rearTyre.offset,
    y: CAR.rearAxleY,
    labelX: 272,
    labelY: 262,
    anchor: 'end',
  },
]

/**
 * The car itself. No data in here at all: every shape is fixed geometry, and
 * the tyres are drawn separately on top so the only coloured thing on screen is
 * the thing being measured.
 */
function CarBody() {
  const line = 'var(--color-line-bright)'
  const panel = 'var(--color-card)'
  const faint = 'var(--color-line)'

  return (
    <g fill={panel} stroke={line} strokeWidth="1.1" strokeLinejoin="round">
      {/* Front wing. As wide as the car and no wider, with the endplates
          turning down at each tip. */}
      <path d="M 95 18 L 185 18 Q 140 25 95 18 Z" fill={panel} />
      <path d="M 95 18 L 185 18 L 185 30 Q 140 36 95 30 Z" />
      <rect x="92" y="15" width="5" height="20" rx="1.5" />
      <rect x="183" y="15" width="5" height="20" rx="1.5" />
      <path d="M 104 27 Q 140 32 176 27" fill="none" stroke={faint} strokeWidth="0.7" />

      {/* Nose. Narrow at the tip and tapering back into the chassis, joined to
          the wing rather than floating above it. */}
      <path d="M 134 27 Q 140 24 146 27 L 152 68 L 156 104 L 124 104 L 128 68 Z" />

      {/* Chassis, cockpit opening and the halo hoop over it. */}
      <path d="M 124 104 L 156 104 L 158 132 L 122 132 Z" />
      <path
        d="M 128 108 L 152 108 L 150 128 Q 140 132 130 128 Z"
        fill="var(--color-ground)"
        stroke={line}
      />
      <path d="M 130 116 Q 140 106 150 116" fill="none" stroke={line} strokeWidth="1.5" />
      <path d="M 140 106 L 140 112" stroke={line} strokeWidth="1.2" />

      {/* Sidepods. Widest just behind the cockpit, then the coke-bottle taper
          in towards the rear axle. The notches are the radiator inlets. */}
      <path d="M 123 132 L 157 132 Q 165 137 165 150 L 165 178 Q 165 196 154 210 L 126 210 Q 115 196 115 178 L 115 150 Q 115 137 123 132 Z" />
      <path d="M 115 139 L 115 154" fill="none" stroke={faint} strokeWidth="2.6" />
      <path d="M 165 139 L 165 154" fill="none" stroke={faint} strokeWidth="2.6" />

      {/* Engine cover along the spine, tapering to the rear. */}
      <path
        d="M 131 134 L 149 134 L 146 208 L 134 208 Z"
        fill="var(--color-raised)"
        stroke={faint}
      />

      {/* Floor and diffuser, narrowing to the rear crash structure. */}
      <path d="M 126 210 L 154 210 L 149 248 L 131 248 Z" />
      <g stroke={faint} strokeWidth="0.7" fill="none">
        <path d="M 133 238 L 147 238" />
        <path d="M 134 243 L 146 243" />
      </g>

      {/* Suspension: two wishbones per corner, reaching to the wheel centres. */}
      <g stroke={line} strokeWidth="1.3" fill="none">
        <path d="M 127 70 L 106 74" />
        <path d="M 127 92 L 106 84" />
        <path d="M 153 70 L 174 74" />
        <path d="M 153 92 L 174 84" />
        <path d="M 120 228 L 104 238" />
        <path d="M 122 248 L 104 250" />
        <path d="M 160 228 L 176 238" />
        <path d="M 158 248 L 176 250" />
      </g>

      {/* Rear wing. Much narrower than the track, which is the detail that
          stops a top-down F1 car reading as a plane. Carried on two pylons off
          the crash structure rather than floating behind it. */}
      <path d="M 136 246 L 144 246 L 143 258 L 137 258 Z" stroke={faint} />
      <rect x="114" y="254" width="5" height="20" rx="1.5" />
      <rect x="161" y="254" width="5" height="20" rx="1.5" />
      <path d="M 117 257 L 163 257 L 163 265 L 117 265 Z" />
      <path d="M 117 268 L 163 268 L 163 273 L 117 273 Z" fill="var(--color-raised)" stroke={faint} />
    </g>
  )
}

/**
 * @param energy Per-corner energy share. Values are relative; they are
 *   normalised here so the drawing reads the same whatever units arrive.
 * @param health 0-100 tyre health, used for the fill colour.
 * @param compound Compound in use, for the band colour.
 */
export function TyreTwin({
  energy,
  health,
  performanceLost,
  compound,
  ageLaps,
}: {
  energy: CornerLoad
  /** Null while the timeline is still loading. Never defaulted to 100 -- showing
   *  a fresh-tyre reading for a 40-lap-old set is worse than showing nothing. */
  health: number | null
  performanceLost: number | null
  compound: string
  ageLaps: number
}) {
  const values = Object.values(energy)
  const max = Math.max(...values, 1e-9)
  const min = Math.min(...values)
  const span = Math.max(max - min, 1e-9)

  const band = compoundColour(compound)
  const total = values.reduce((a, b) => a + b, 0) || 1
  const leftShare = (energy.FL + energy.RL) / total
  const frontShare = (energy.FL + energy.FR) / total

  return (
    <div className="flex flex-col items-center gap-3 sm:flex-row sm:items-start sm:gap-6">
      <svg
        viewBox="0 0 280 300"
        className="h-[300px] w-[280px] shrink-0"
        role="img"
        aria-label={`Tyre loading diagram: left side carries ${(leftShare * 100).toFixed(0)} percent of the energy`}
      >
        <CarBody />

        {CORNERS.map(({ key, x, y, labelX, labelY, anchor }) => {
          const value = energy[key]
          // Intensity is relative WITHIN this car, so the hardest-worked corner
          // always reads as the hardest-worked corner regardless of absolute scale.
          const intensity = (value - min) / span
          const share = (100 * value) / total
          const isFront = key === 'FL' || key === 'FR'
          const { width, height } = isFront ? CAR.frontTyre : CAR.rearTyre
          const barWidth = 40
          const barX = anchor === 'start' ? labelX : labelX - barWidth

          return (
            <g key={key}>
              {/* The tyre. Filled towards the compound colour by how hard this
                  corner is worked, so the asymmetry is visible before any
                  number is read. */}
              <rect
                x={x - width / 2}
                y={y - height / 2}
                width={width}
                height={height}
                rx={width / 2.6}
                fill={`color-mix(in oklab, ${band} ${16 + intensity * 68}%, var(--color-ground))`}
                stroke={band}
                strokeWidth="1.6"
              />
              {/* Tread grooves, purely so it reads as a tyre and not a pill. */}
              <g stroke={band} strokeWidth="0.7" opacity="0.45">
                <line x1={x - width / 4} y1={y - height / 2 + 5} x2={x - width / 4} y2={y + height / 2 - 5} />
                <line x1={x + width / 4} y1={y - height / 2 + 5} x2={x + width / 4} y2={y + height / 2 - 5} />
              </g>

              {/* Readout, out in the margin beside its own corner. */}
              <text
                x={labelX}
                y={labelY}
                textAnchor={anchor}
                className="num"
                style={{ fontSize: 19, fontWeight: 600, fill: 'var(--color-ink)' }}
              >
                {share.toFixed(0)}
                <tspan style={{ fontSize: 11, fill: 'var(--color-ink-faint)' }}>%</tspan>
              </text>
              <text
                x={labelX}
                y={labelY + 13}
                textAnchor={anchor}
                style={{ fontSize: 9, fill: 'var(--color-ink-faint)', letterSpacing: '0.08em' }}
              >
                {key}
              </text>
              {/* A bar on the same scale for all four, so they compare by eye. */}
              <rect x={barX} y={labelY + 19} width={barWidth} height="3" rx="1.5" fill="var(--color-raised)" />
              <rect
                x={anchor === 'start' ? barX : barX + barWidth * (1 - share / 50)}
                y={labelY + 19}
                width={barWidth * Math.min(share / 50, 1)}
                height="3"
                rx="1.5"
                fill={band}
              />
            </g>
          )
        })}

        <text
          x="140"
          y="298"
          textAnchor="middle"
          style={{ fontSize: 9, fill: 'var(--color-ink-faint)' }}
        >
          share of frictional energy per corner
        </text>
      </svg>

      <div className="min-w-0 flex-1 space-y-3">
        <div>
          <div className="text-[11px] text-ink-faint">Tyre health</div>
          {health == null ? (
            <div className="num text-[34px] leading-none font-medium text-ink-faint">--</div>
          ) : (
            <>
              <div className="flex items-baseline gap-2">
                <span
                  className="num text-[34px] leading-none font-medium"
                  style={{ color: healthColour(health) }}
                >
                  {health.toFixed(0)}
                </span>
                <span className="text-[12px] text-ink-faint">/ 100</span>
              </div>
              {performanceLost != null && (
                <div className="num mt-0.5 text-[11px] text-ink-faint">
                  {performanceLost.toFixed(2)} s/lap slower than a fresh set
                </div>
              )}
              <div className="mt-1.5 h-1.5 w-full max-w-[240px] overflow-hidden rounded-full bg-raised">
                <div
                  className="h-full rounded-full transition-[width] duration-500"
                  style={{ width: `${health}%`, background: healthColour(health) }}
                />
              </div>
            </>
          )}
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-[11.5px]">
          <div>
            <dt className="text-ink-faint">Compound</dt>
            <dd className="flex items-center gap-1.5 text-ink">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: band }}
              />
              {compound}
            </dd>
          </div>
          <div>
            <dt className="text-ink-faint">Age</dt>
            <dd className="num text-ink">{ageLaps.toFixed(0)} laps</dd>
          </div>
          <div>
            <dt className="text-ink-faint">Left / right split</dt>
            <dd className="num text-ink">
              {(leftShare * 100).toFixed(0)} / {((1 - leftShare) * 100).toFixed(0)}
            </dd>
          </div>
          <div>
            <dt className="text-ink-faint">Front / rear split</dt>
            <dd className="num text-ink">
              {(frontShare * 100).toFixed(0)} / {((1 - frontShare) * 100).toFixed(0)}
            </dd>
          </div>
        </dl>

        <p className="max-w-[46ch] text-[11px] leading-relaxed text-ink-faint">
          Corner loading comes from the racing line: curvature drives lateral
          acceleration, which shifts load to the outside of the car, so a
          mostly right-hand circuit works the left tyres hardest.
        </p>
      </div>
    </div>
  )
}

/**
 * Per-corner energy shares come from the API, which reads them from the recorded
 * physics validation rather than reconstructing them.
 *
 * The dashboard runs from cached timing data, which does not carry the position
 * telemetry the physics layer needs -- that is a far larger download, used by the
 * experiments. So the twin shows measured values for circuits that have been
 * analysed and says plainly when a circuit has not, rather than filling the gap
 * with an even split presented as a result.
 */
export interface CornerEnergyResult {
  circuit: string
  measured: boolean
  reason?: string
  corner_share?: CornerLoad
  left_side_energy_share?: number
  front_axle_energy_share?: number
  peak_lateral_g?: number
  published_direction?: string
  predicted_direction?: string
  n_laps?: number
}

export const EVEN_SPLIT: CornerLoad = { FL: 0.25, FR: 0.25, RL: 0.25, RR: 0.25 }
