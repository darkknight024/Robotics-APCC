import { ChevronDown } from 'lucide-react'
import { useState } from 'react'

export function PlotGroup({
  title,
  defaultOpen = true,
  children,
}: {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border border-border rounded-md overflow-hidden mb-2">
      <button
        type="button"
        className="w-full flex items-center justify-between px-2 py-1.5 bg-surface-2 text-xxs font-medium text-text-secondary uppercase tracking-wider"
        onClick={() => setOpen((o) => !o)}
      >
        {title}
        <ChevronDown className={`w-3.5 h-3.5 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && <div className="p-2 space-y-2">{children}</div>}
    </div>
  )
}
