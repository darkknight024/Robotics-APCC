import { useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Activity, Gamepad2, Bot } from 'lucide-react'

export function AppShell({ children }: { children: React.ReactNode }) {
  const location = useLocation()
  const currentTab = location.pathname === '/teleop' ? 'teleop' : 'analysis'

  useEffect(() => {
    document.title = `APCC Visualizer — ${currentTab === 'analysis' ? 'Analysis' : 'TeleOp'}`
  }, [currentTab])

  return (
    <div className="h-screen flex flex-col bg-surface-0 overflow-hidden">
      {/* Top Navigation Bar */}
      <header className="h-11 flex-shrink-0 bg-surface-1 border-b border-border flex items-center px-4 gap-6">
        {/* Logo / Title */}
        <div className="flex items-center gap-2 mr-4">
          <Bot className="w-5 h-5 text-accent-blue" />
          <span className="text-sm font-semibold text-text-primary tracking-tight">
            Robotics-APCC
          </span>
          <span className="text-xxs font-mono text-text-muted ml-1">
            VISUALIZER
          </span>
        </div>

        {/* Tab Navigation */}
        <nav className="flex items-center gap-1 h-full">
          <Link
            to="/analysis"
            className={`
              flex items-center gap-1.5 px-3 h-full text-xs font-medium transition-colors
              ${currentTab === 'analysis' ? 'tab-active' : 'tab-inactive'}
            `}
          >
            <Activity className="w-3.5 h-3.5" />
            Analysis
          </Link>
          <Link
            to="/teleop"
            className={`
              flex items-center gap-1.5 px-3 h-full text-xs font-medium transition-colors
              ${currentTab === 'teleop' ? 'tab-active' : 'tab-inactive'}
            `}
          >
            <Gamepad2 className="w-3.5 h-3.5" />
            TeleOp
          </Link>
        </nav>

        {/* Right side spacer / future controls */}
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse" title="Backend connected" />
          <span className="text-xxs text-text-muted font-mono">v0.1</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  )
}
