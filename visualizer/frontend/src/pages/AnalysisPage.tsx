import { ResizableLayout } from '../components/layout/ResizableLayout'
import { ViserEmbed } from '../components/viewer/ViserEmbed'
import { RobotSelector } from '../components/analysis/RobotSelector'
import { Upload, Database, Settings2, Play } from 'lucide-react'
import { useAnalysisStore } from '../stores/analysisStore'

function LeftPanel() {
  const { currentStep } = useAnalysisStore()

  return (
    <div className="flex flex-col h-full">
      {/* Panel Header */}
      <div className="panel-header flex items-center justify-between">
        <span>Workflow</span>
        <span className="text-xxs font-mono text-text-muted">{currentStep}</span>
      </div>

      {/* Step Indicators */}
      <div className="px-3 py-2 border-b border-border">
        <div className="flex items-center gap-1">
          {[
            { step: 'upload', icon: Upload, label: 'Upload' },
            { step: 'robot', icon: Settings2, label: 'Robot' },
            { step: 'run', icon: Play, label: 'Run' },
            { step: 'results', icon: Database, label: 'Results' },
          ].map(({ step, icon: Icon, label }, idx) => (
            <div key={step} className="flex items-center">
              {idx > 0 && <div className="w-4 h-px bg-border mx-0.5" />}
              <div className={`
                flex items-center gap-1 px-1.5 py-0.5 rounded text-xxs
                ${currentStep === step
                  ? 'bg-accent-blue/15 text-accent-blue'
                  : 'text-text-muted'
                }
              `}>
                <Icon className="w-3 h-3" />
                <span className="hidden xl:inline">{label}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Robot Selector (always visible) */}
      <div className="border-b border-border">
        <div className="panel-header">Robot</div>
        <RobotSelector />
      </div>

      {/* Future: Upload, Frame Config, etc. will go here */}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="text-center space-y-2">
          <Upload className="w-8 h-8 text-text-muted mx-auto" />
          <p className="text-xs text-text-muted">
            Upload a CSV file to begin analysis
          </p>
          <p className="text-xxs text-text-muted">
            Drag & drop or click to browse
          </p>
        </div>
      </div>
    </div>
  )
}

function RightPanel() {
  return (
    <div className="flex flex-col h-full">
      <div className="panel-header">Plots & Data</div>
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="text-center space-y-2">
          <Database className="w-8 h-8 text-text-muted mx-auto" />
          <p className="text-xs text-text-muted">
            Run an analysis to see plots here
          </p>
          <p className="text-xxs text-text-muted">
            Joint angles, singularity, manipulability, TOPP-RA, ECFX
          </p>
        </div>
      </div>
    </div>
  )
}

export function AnalysisPage() {
  return (
    <ResizableLayout
      leftPanel={<LeftPanel />}
      centerPanel={<ViserEmbed />}
      rightPanel={<RightPanel />}
    />
  )
}
