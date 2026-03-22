import { ResizableLayout } from '../components/layout/ResizableLayout'
import { ViserEmbed } from '../components/viewer/ViserEmbed'
import { UploadStep } from '../components/analysis/UploadStep'
import { DetectStep } from '../components/analysis/DetectStep'
import { FrameStep } from '../components/analysis/FrameStep'
import { RobotStep } from '../components/analysis/RobotStep'
import { Database, Settings2, Crosshair, Bot, Upload } from 'lucide-react'
import { useAnalysisStore } from '../stores/analysisStore'
import type { AnalysisStep } from '../types/data'

function LeftPanel() {
  const { currentStep, detectionResult } = useAnalysisStore()

  const steps: { step: AnalysisStep; label: string; icon: typeof Upload }[] = [
    { step: 'upload', label: 'Upload', icon: Upload },
    { step: 'detect', label: 'Detect', icon: Database },
    ...(detectionResult?.has_task_space
      ? [{ step: 'frame' as const, label: 'Frame', icon: Crosshair }]
      : []),
    { step: 'robot', label: 'Robot', icon: Bot },
  ]

  const renderWorkflow = () => {
    switch (currentStep) {
      case 'upload':
        return <UploadStep />
      case 'detect':
        return <DetectStep />
      case 'frame':
        return detectionResult?.has_task_space ? <FrameStep /> : <DetectStep />
      case 'robot':
        return <RobotStep />
      default:
        return (
          <div className="p-4 text-xs text-text-muted">
            <Settings2 className="w-6 h-6 mx-auto mb-2 opacity-50" />
            This step is not part of Phase 2 workflow yet.
          </div>
        )
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="panel-header flex items-center justify-between">
        <span>Workflow</span>
        <span className="text-xxs font-mono text-text-muted">{currentStep}</span>
      </div>

      <div className="px-3 py-2 border-b border-border">
        <div className="flex flex-wrap items-center gap-1">
          {steps.map(({ step, icon: Icon, label }, idx) => (
            <div key={step} className="flex items-center">
              {idx > 0 && <div className="w-3 h-px bg-border mx-0.5" />}
              <div
                className={`
                flex items-center gap-1 px-1.5 py-0.5 rounded text-xxs
                ${currentStep === step ? 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted'}
              `}
              >
                <Icon className="w-3 h-3" />
                <span className="hidden xl:inline">{label}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0">{renderWorkflow()}</div>
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
          <p className="text-xs text-text-muted">Run an analysis to see plots here</p>
          <p className="text-xxs text-text-muted">Phase 3: IK/FK runs and Plotly panels</p>
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
