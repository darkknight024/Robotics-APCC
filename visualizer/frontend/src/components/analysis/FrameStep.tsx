import { useEffect } from 'react'
import { toast } from 'sonner'
import { fetchKnives } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'

export function FrameStep() {
  const {
    knives,
    setKnives,
    useBaseFrame,
    setUseBaseFrame,
    selectedKnife,
    setSelectedKnife,
    setStep,
  } = useAnalysisStore()

  useEffect(() => {
    void (async () => {
      const json = await fetchKnives()
      if (json.ok && json.data?.length) {
        setKnives(json.data)
        const current = useAnalysisStore.getState().selectedKnife
        if (!current) {
          setSelectedKnife(json.data[0].name)
        }
      } else if (!json.ok) {
        toast.error(json.error || 'Failed to load knives')
      }
    })()
  }, [setKnives, setSelectedKnife])

  return (
    <div className="p-3 space-y-3 text-xs">
      <p className="text-text-muted text-xxs">
        Is your TCP pose already expressed in the robot base frame, or in the knife (T_P_K) frame?
      </p>

      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="radio"
          name="frame"
          checked={useBaseFrame}
          onChange={() => setUseBaseFrame(true)}
          className="accent-accent-blue"
        />
        <span>Already in base frame (T_B_P)</span>
      </label>

      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="radio"
          name="frame"
          checked={!useBaseFrame}
          onChange={() => setUseBaseFrame(false)}
          className="accent-accent-blue"
        />
        <span>In knife frame — transform using knife pose</span>
      </label>

      {!useBaseFrame && (
        <div className="space-y-1">
          <label className="text-xxs text-text-muted uppercase tracking-wider">Knife pose</label>
          <select
            className="select-field w-full text-xs"
            value={selectedKnife || ''}
            onChange={(e) => setSelectedKnife(e.target.value || null)}
          >
            {knives.map((k) => (
              <option key={k.name} value={k.name}>
                {k.name} — {k.description}
              </option>
            ))}
          </select>
        </div>
      )}

      <button type="button" className="btn-primary text-xxs w-full py-2" onClick={() => setStep('robot')}>
        Continue to robot
      </button>
    </div>
  )
}
