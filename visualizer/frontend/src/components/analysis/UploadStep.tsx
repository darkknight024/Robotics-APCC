import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import Papa from 'papaparse'
import { Upload, FileSpreadsheet, Loader2 } from 'lucide-react'
import { toast } from 'sonner'
import { uploadCsv } from '../../lib/api'
import { useAnalysisStore } from '../../stores/analysisStore'

function PreviewTable({ rows }: { rows: string[][] }) {
  if (rows.length === 0) return null
  const head = rows.slice(0, 12)
  return (
    <div className="mt-3 overflow-x-auto rounded border border-border max-h-40 overflow-y-auto">
      <table className="w-full text-xxs font-mono">
        <tbody>
          {head.map((row, i) => (
            <tr key={i} className="border-b border-border/50">
              {row.map((c, j) => (
                <td key={j} className="px-1.5 py-0.5 whitespace-nowrap text-text-muted">
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function UploadStep() {
  const { setSessionId, setStep } = useAnalysisStore()
  const [busy, setBusy] = useState(false)
  const [localPreview, setLocalPreview] = useState<string[][]>([])

  const onDrop = useCallback(
    async (files: File[]) => {
      const file = files[0]
      if (!file) return
      setBusy(true)
      Papa.parse(file, {
        preview: 15,
        complete: (res) => {
          const data = (res.data as string[][]).filter((r) => r.some((c) => String(c).trim()))
          setLocalPreview(data)
        },
      })
      const json = await uploadCsv(file)
      setBusy(false)
      if (!json.ok || !json.data) {
        toast.error(json.error || 'Upload failed')
        return
      }
      setSessionId(json.data.session_id)
      toast.success('File uploaded')
      setStep('detect')
    },
    [setSessionId, setStep],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxFiles: 1,
    disabled: busy,
  })

  return (
    <div className="p-3 space-y-3">
      <div
        {...getRootProps()}
        className={`
          border border-dashed rounded-md px-4 py-8 text-center cursor-pointer transition-colors
          ${isDragActive ? 'border-accent-blue bg-accent-blue/10' : 'border-border hover:border-text-muted'}
          ${busy ? 'opacity-50 pointer-events-none' : ''}
        `}
      >
        <input {...getInputProps()} />
        {busy ? (
          <Loader2 className="w-8 h-8 text-accent-blue mx-auto animate-spin" />
        ) : (
          <Upload className="w-8 h-8 text-text-muted mx-auto mb-2" />
        )}
        <p className="text-xs text-text-primary">Drop a CSV here, or click to browse</p>
        <p className="text-xxs text-text-muted mt-1">Toolpath or RobotStudio-style CSV</p>
      </div>
      {localPreview.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 text-xxs text-text-muted">
            <FileSpreadsheet className="w-3 h-3" />
            Local preview (first rows)
          </div>
          <PreviewTable rows={localPreview} />
        </div>
      )}
    </div>
  )
}
