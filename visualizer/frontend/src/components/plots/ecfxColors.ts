/** Discrete cf quadrant colors (README §14.3). */
export function ecfxDiscreteColor(cf: number): string {
  switch (cf) {
    case -2:
      return '#3b82f6'
    case -1:
      return '#22d3ee'
    case 0:
      return '#22c55e'
    case 1:
      return '#f97316'
    case 2:
      return '#ef4444'
    default:
      return '#94a3b8'
  }
}
