import { useEffect, useCallback } from 'react'
import { ChevronDown, Loader2, Info } from 'lucide-react'
import { useAnalysisStore } from '../../stores/analysisStore'
import type { RobotOption } from '../../types/data'

const API_BASE = 'http://localhost:8080'

export function RobotSelector() {
  const { robots, setRobots, selectedRobot, setSelectedRobot } = useAnalysisStore()

  const fetchRobots = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/robots`)
      const json = await res.json()
      if (json.ok && json.data) {
        setRobots(json.data)
        if (json.data.length > 0 && !selectedRobot) {
          setSelectedRobot(json.data[0].name)
          // Notify backend to load this robot in Viser
          loadRobotInViser(json.data[0].name)
        }
      }
    } catch (err) {
      console.error('Failed to fetch robots:', err)
    }
  }, [setRobots, selectedRobot, setSelectedRobot])

  useEffect(() => {
    fetchRobots()
  }, [fetchRobots])

  const loadRobotInViser = async (robotName: string) => {
    try {
      await fetch(`${API_BASE}/api/load-robot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ robot_name: robotName }),
      })
    } catch (err) {
      console.error('Failed to load robot:', err)
    }
  }

  const handleRobotChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const name = e.target.value
    setSelectedRobot(name)
    loadRobotInViser(name)
  }

  const selected: RobotOption | undefined = robots.find(r => r.name === selectedRobot)

  if (robots.length === 0) {
    return (
      <div className="px-3 py-4 flex items-center gap-2 text-text-muted text-xs">
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
        Loading robots...
      </div>
    )
  }

  return (
    <div className="px-3 py-3 space-y-3">
      {/* Dropdown */}
      <div className="space-y-1.5">
        <label className="text-xxs font-medium uppercase tracking-wider text-text-muted">
          Robot Model
        </label>
        <div className="relative">
          <select
            value={selectedRobot || ''}
            onChange={handleRobotChange}
            className="select-field w-full pr-8"
          >
            {robots.map(robot => (
              <option key={robot.name} value={robot.name}>
                {robot.name}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-muted pointer-events-none" />
        </div>
      </div>

      {/* Robot Info Card */}
      {selected && (
        <div className="panel p-2.5 space-y-2">
          <div className="flex items-center gap-1.5 text-xxs text-text-muted">
            <Info className="w-3 h-3" />
            <span>{selected.description}</span>
          </div>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xxs">
            <div className="text-text-muted">Reach</div>
            <div className="font-mono text-text-primary" data-numeric>
              {(selected.reach_m * 1000).toFixed(0)} mm
            </div>
            <div className="text-text-muted">Payload</div>
            <div className="font-mono text-text-primary" data-numeric>
              {selected.payload_kg} kg
            </div>
          </div>

          {/* Joint velocity limits */}
          <div className="space-y-1 pt-1 border-t border-border">
            <div className="text-xxs text-text-muted">Velocity limits (rad/s)</div>
            <div className="grid grid-cols-6 gap-1">
              {selected.velocity_limits_rad_s.map((v, i) => (
                <div key={i} className="text-center">
                  <div className="text-xxs text-text-muted">J{i + 1}</div>
                  <div className="text-xxs font-mono text-text-primary" data-numeric>
                    {v.toFixed(1)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
