import { ResizableLayout } from '../components/layout/ResizableLayout'
import { ViserEmbed } from '../components/viewer/ViserEmbed'
import { TimelineBar } from '../components/viewer/TimelineBar'
import { UploadStep } from '../components/analysis/UploadStep'
import { DetectStep } from '../components/analysis/DetectStep'
import { FrameStep } from '../components/analysis/FrameStep'
import { RobotStep } from '../components/analysis/RobotStep'
import { ActionStep } from '../components/analysis/ActionStep'
import { ConfigPanel } from '../components/analysis/ConfigPanel'
import { RunPanel } from '../components/analysis/RunPanel'
import { ResultsStep } from '../components/analysis/ResultsStep'
import { KinematicsPlotDashboard } from '../components/plots/KinematicsPlotDashboard'
import {
  Database,
  Settings2,
  Crosshair,
  Bot,
  Upload,
  Zap,
  Sliders,
  Play,
  CheckCircle,
} from 'lucide-react'
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
    { step: 'action', label: 'Mode', icon: Zap },
    { step: 'config', label: 'Solver', icon: Sliders },
    { step: 'run', label: 'Run', icon: Play },
    { step: 'results', label: 'Done', icon: CheckCircle },
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
      case 'action':
        return <ActionStep />
      case 'config':
        return <ConfigPanel />
      case 'run':
        return <RunPanel />
      case 'results':
        return <ResultsStep />
      default:
        return (
          <div className="p-4 text-xs text-text-muted">
            <Settings2 className="w-6 h-6 mx-auto mb-2 opacity-50" />
            Unknown step
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

function CenterPanel() {
  const runResult = useAnalysisStore((s) => s.runResult)
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex-1 min-h-0">
        <ViserEmbed />
      </div>
      {runResult ? <TimelineBar /> : null}
    </div>
  )
}

function RightPanel() {
  const runResult = useAnalysisStore((s) => s.runResult)
  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="panel-header">Plots & Data</div>
      <div className="flex-1 overflow-y-auto min-h-0 p-2">
        {runResult ? (
          <KinematicsPlotDashboard result={runResult} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full min-h-[120px] p-4 text-center space-y-2">
            <Database className="w-8 h-8 text-text-muted mx-auto" />
            <p className="text-xs text-text-muted">Run IK or FK to see kinematics plots</p>
          </div>
        )}
      </div>
    </div>
  )
}

export function AnalysisPage() {
  return (
    <ResizableLayout leftPanel={<LeftPanel />} centerPanel={<CenterPanel />} rightPanel={<RightPanel />} />
  )
}
