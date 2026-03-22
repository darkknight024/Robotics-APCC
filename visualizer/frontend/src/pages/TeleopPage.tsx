import { ViserEmbed } from '../components/viewer/ViserEmbed'
import { Gamepad2, Keyboard, Circle } from 'lucide-react'
import { useTeleopStore } from '../stores/teleopStore'

export function TeleopPage() {
  const { mode, isConnected } = useTeleopStore()

  return (
    <div className="h-full flex">
      {/* Left Control Panel */}
      <div className="w-72 flex-shrink-0 bg-surface-1 border-r border-border overflow-y-auto">
        <div className="panel-header flex items-center justify-between">
          <span>TeleOp Controls</span>
          <div className="flex items-center gap-1.5">
            <Circle className={`w-2 h-2 ${isConnected ? 'fill-accent-green text-accent-green' : 'fill-accent-red text-accent-red'}`} />
            <span className="text-xxs font-mono text-text-muted">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Mode Selection */}
        <div className="px-3 py-3 border-b border-border space-y-2">
          <label className="text-xxs font-medium uppercase tracking-wider text-text-muted">
            Control Mode
          </label>
          <div className="grid grid-cols-2 gap-1">
            <button className={`
              px-2 py-2 rounded text-xs font-medium transition-colors
              ${mode === 'task_space'
                ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/30'
                : 'bg-surface-2 text-text-secondary border border-border hover:bg-surface-3'
              }
            `}>
              Task Space
            </button>
            <button className={`
              px-2 py-2 rounded text-xs font-medium transition-colors
              ${mode === 'joint_space'
                ? 'bg-accent-blue/15 text-accent-blue border border-accent-blue/30'
                : 'bg-surface-2 text-text-secondary border border-border hover:bg-surface-3'
              }
            `}>
              Joint Space
            </button>
          </div>
        </div>

        {/* Keyboard Map Preview */}
        <div className="px-3 py-3 border-b border-border space-y-2">
          <div className="flex items-center gap-1.5">
            <Keyboard className="w-3.5 h-3.5 text-text-muted" />
            <label className="text-xxs font-medium uppercase tracking-wider text-text-muted">
              Key Bindings
            </label>
          </div>
          <div className="grid grid-cols-3 gap-1 text-center">
            {[
              ['', 'W', '', '+Z'],
              ['A', 'S', 'D', '−X, −Z, +X'],
              ['', '', '', ''],
              ['Q', '', 'E', '+Y, , −Y'],
            ].map((row, ri) => (
              <div key={ri} className="contents">
                {row.slice(0, 3).map((key, ki) => (
                  <div key={ki} className={`
                    h-7 rounded text-xxs font-mono flex items-center justify-center
                    ${key ? 'bg-surface-3 text-text-primary border border-border' : ''}
                  `}>
                    {key}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <p className="text-xxs text-text-muted">
            Space to record waypoint • R to auto-record • Tab to switch mode
          </p>
        </div>

        {/* Placeholder for metrics HUD */}
        <div className="px-3 py-3 space-y-2">
          <label className="text-xxs font-medium uppercase tracking-wider text-text-muted">
            Live Metrics
          </label>
          <div className="space-y-1.5 text-xs text-text-muted">
            <p>Connect to backend to see live metrics</p>
          </div>
        </div>
      </div>

      {/* Center — 3D Viewer */}
      <div className="flex-1 relative">
        <ViserEmbed />

        {/* Floating TeleOp indicator */}
        <div className="absolute top-3 left-3 z-10 flex items-center gap-2 px-3 py-1.5 rounded bg-surface-3/80 backdrop-blur-sm">
          <Gamepad2 className="w-4 h-4 text-accent-cyan" />
          <span className="text-xs font-medium text-text-primary">TeleOp Mode</span>
          <span className="text-xxs font-mono text-text-muted">
            {mode === 'task_space' ? 'Task Space' : 'Joint Space'}
          </span>
        </div>
      </div>
    </div>
  )
}
