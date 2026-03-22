import { Panel, Group, Separator } from 'react-resizable-panels'

interface ResizableLayoutProps {
  leftPanel: React.ReactNode
  centerPanel: React.ReactNode
  rightPanel: React.ReactNode
}

export function ResizableLayout({ leftPanel, centerPanel, rightPanel }: ResizableLayoutProps) {
  return (
    <Group orientation="horizontal" className="h-full">
      {/* Left Panel — Config / Workflow */}
      <Panel defaultSize="22%" minSize="14%" maxSize="40%">
        <div className="h-full bg-surface-1 overflow-y-auto">
          {leftPanel}
        </div>
      </Panel>

      <Separator />

      {/* Center Panel — 3D Viewer */}
      <Panel defaultSize="48%" minSize="25%">
        <div className="h-full bg-surface-0 overflow-hidden">
          {centerPanel}
        </div>
      </Panel>

      <Separator />

      {/* Right Panel — Plots / Metrics */}
      <Panel defaultSize="30%" minSize="14%" maxSize="50%">
        <div className="h-full bg-surface-1 overflow-y-auto">
          {rightPanel}
        </div>
      </Panel>
    </Group>
  )
}
