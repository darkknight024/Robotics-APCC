import { useState } from 'react'
import { Maximize2, Minimize2, RefreshCw } from 'lucide-react'

const VISER_URL = 'http://localhost:8081'

export function ViserEmbed() {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [key, setKey] = useState(0) // force iframe reload

  const handleReload = () => setKey(prev => prev + 1)

  return (
    <div className={`relative w-full h-full ${isFullscreen ? 'fixed inset-0 z-50 bg-surface-0' : ''}`}>
      {/* Toolbar */}
      <div className="absolute top-2 right-2 z-10 flex items-center gap-1">
        <button
          onClick={handleReload}
          className="p-1.5 rounded bg-surface-3/80 backdrop-blur-sm hover:bg-surface-4 transition-colors"
          title="Reload 3D viewer"
        >
          <RefreshCw className="w-3.5 h-3.5 text-text-secondary" />
        </button>
        <button
          onClick={() => setIsFullscreen(!isFullscreen)}
          className="p-1.5 rounded bg-surface-3/80 backdrop-blur-sm hover:bg-surface-4 transition-colors"
          title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {isFullscreen
            ? <Minimize2 className="w-3.5 h-3.5 text-text-secondary" />
            : <Maximize2 className="w-3.5 h-3.5 text-text-secondary" />
          }
        </button>
      </div>

      {/* Viser iframe */}
      <iframe
        key={key}
        src={VISER_URL}
        className="w-full h-full border-0"
        title="3D Robot Viewer"
        allow="autoplay; fullscreen"
      />

      {/* Connection status overlay */}
      <div className="absolute bottom-2 left-2 z-10">
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-surface-3/80 backdrop-blur-sm">
          <div className="w-1.5 h-1.5 rounded-full bg-accent-green animate-pulse" />
          <span className="text-xxs font-mono text-text-muted">Viser :8081</span>
        </div>
      </div>
    </div>
  )
}
