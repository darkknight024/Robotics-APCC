import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'sonner'
import { AppShell } from './components/layout/AppShell'
import { AnalysisPage } from './pages/AnalysisPage'
import { TeleopPage } from './pages/TeleopPage'

function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/teleop" element={<TeleopPage />} />
          <Route path="*" element={<Navigate to="/analysis" replace />} />
        </Routes>
      </AppShell>
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#1a1a24',
            border: '1px solid #2d2d3d',
            color: '#e4e4ef',
            fontSize: '0.8rem',
          },
        }}
      />
    </BrowserRouter>
  )
}

export default App
