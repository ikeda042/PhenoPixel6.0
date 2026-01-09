import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ChakraProvider } from '@chakra-ui/react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import '@fontsource/noto-sans-jp/400.css'
import '@fontsource/noto-sans-jp/600.css'
import '@fontsource/noto-sans-jp/700.css'
import './index.css'
import App from './App'
import AnnotationPage from './pages/AnnotationPage'
import BulkEnginePage from './pages/BulkEnginePage'
import CellExtractionPage from './pages/CellExtractionPage'
import CellsPage from './pages/CellsPage'
import DatabasesPage from './pages/DatabasesPage'
import FilesPage from './pages/FilesPage'
import Nd2ParserPage from './pages/Nd2ParserPage'
import Nd2FilesPage from './pages/Nd2FilesPage'
import TopPage from './pages/TopPage'
import system from './theme'
import { applyThemeMode, getInitialTheme } from './utils/themeMode'

applyThemeMode(getInitialTheme())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ChakraProvider value={system}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<TopPage />} />
          <Route path="/annotation" element={<AnnotationPage />} />
          <Route path="/bulk-engine" element={<BulkEnginePage />} />
          <Route path="/cell-extraction" element={<CellExtractionPage />} />
          <Route path="/cells" element={<CellsPage />} />
          <Route path="/databases" element={<DatabasesPage />} />
          <Route path="/files" element={<FilesPage />} />
          <Route path="/nd2files" element={<Nd2FilesPage />} />
          <Route path="/nd2parser" element={<Nd2ParserPage />} />
          <Route path="/docs" element={<App />} />
        </Routes>
      </BrowserRouter>
    </ChakraProvider>
  </StrictMode>,
)
