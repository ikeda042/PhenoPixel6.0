import { useEffect, useMemo, useRef, useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router-dom'
import {
  AspectRatio,
  Badge,
  Box,
  BreadcrumbCurrentLink,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbRoot,
  BreadcrumbSeparator,
  Button,
  Container,
  Grid,
  Heading,
  HStack,
  NativeSelect,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react'
import { strFromU8, unzipSync } from 'fflate'
import AppHeader from '../components/AppHeader'
import ReloadButton from '../components/ReloadButton'
import ThemeToggle from '../components/ThemeToggle'
import { getApiBase } from '../utils/apiBase'

type BulkCell = {
  cellId: string
  label: string
  url: string
}

type ManifestEntry = {
  cell_id?: string
  label?: string
  file?: string
}

type CellLengthPair = {
  cell_id: string
  length: number
}

type CellAreaPair = {
  cell_id: string
  area: number
}

type NormalizedMedianPair = {
  cell_id: string
  normalized_median: number
}

type RawIntensityPair = {
  cell_id: string
  intensities: number[]
}

type EntropyMetricsPair = {
  cell_id: string
  entropy_norm: number
  sparsity: number
}

const parseManifestEntries = (payload: unknown): ManifestEntry[] => {
  if (!payload || typeof payload !== 'object') return []
  const cells = (payload as { cells?: unknown }).cells
  if (!Array.isArray(cells)) return []
  return cells.filter((entry) => entry && typeof entry === 'object') as ManifestEntry[]
}

const normalizeLabel = (label: string) => {
  const trimmed = label.trim()
  if (!trimmed) return 'N/A'
  if (trimmed.toUpperCase() === 'N/A') return 'N/A'
  return trimmed
}

const isNumericLabel = (label: string) => /^\d+$/.test(label)

const sortLabels = (labels: string[]) =>
  [...labels].sort((a, b) => {
    if (a === 'N/A') return b === 'N/A' ? 0 : -1
    if (b === 'N/A') return 1
    const aNumeric = isNumericLabel(a)
    const bNumeric = isNumericLabel(b)
    if (aNumeric && bNumeric) return Number(a) - Number(b)
    if (aNumeric) return -1
    if (bNumeric) return 1
    return a.localeCompare(b)
  })

export default function BulkEnginePage() {
  const [searchParams] = useSearchParams()
  const dbName = searchParams.get('dbname') ?? ''
  const apiBase = useMemo(() => getApiBase(), [])
  const bulkZoom = 0.75
  const previewDownscale = 0.5
  const scaledViewportHeight = `calc(100vh / ${bulkZoom})`

  const [cells, setCells] = useState<BulkCell[]>([])
  const [selectedLabel, setSelectedLabel] = useState('1')
  const [selectedChannel, setSelectedChannel] = useState('ph')
  const [isLoading, setIsLoading] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [exportError, setExportError] = useState<string | null>(null)
  const [analysisMode, setAnalysisMode] = useState('cell-length')
  const [analysisLabel, setAnalysisLabel] = useState('1')
  const [analysisChannel, setAnalysisChannel] = useState('ph')
  const [lengthPlotUrl, setLengthPlotUrl] = useState<string | null>(null)
  const [lengthError, setLengthError] = useState<string | null>(null)
  const [isLengthLoading, setIsLengthLoading] = useState(false)
  const [isLengthExporting, setIsLengthExporting] = useState(false)
  const [lengthExportError, setLengthExportError] = useState<string | null>(null)
  const [hasCalculatedLength, setHasCalculatedLength] = useState(false)
  const [areaPlotUrl, setAreaPlotUrl] = useState<string | null>(null)
  const [areaError, setAreaError] = useState<string | null>(null)
  const [isAreaLoading, setIsAreaLoading] = useState(false)
  const [isAreaExporting, setIsAreaExporting] = useState(false)
  const [areaExportError, setAreaExportError] = useState<string | null>(null)
  const [hasCalculatedArea, setHasCalculatedArea] = useState(false)
  const [medianPlotUrl, setMedianPlotUrl] = useState<string | null>(null)
  const [medianError, setMedianError] = useState<string | null>(null)
  const [isMedianLoading, setIsMedianLoading] = useState(false)
  const [isMedianExporting, setIsMedianExporting] = useState(false)
  const [medianExportError, setMedianExportError] = useState<string | null>(null)
  const [hasCalculatedMedian, setHasCalculatedMedian] = useState(false)
  const [entropyPlotUrl, setEntropyPlotUrl] = useState<string | null>(null)
  const [entropyError, setEntropyError] = useState<string | null>(null)
  const [isEntropyLoading, setIsEntropyLoading] = useState(false)
  const [isEntropyExporting, setIsEntropyExporting] = useState(false)
  const [entropyExportError, setEntropyExportError] = useState<string | null>(null)
  const [hasCalculatedEntropy, setHasCalculatedEntropy] = useState(false)
  const [isRawExporting, setIsRawExporting] = useState(false)
  const [rawExportError, setRawExportError] = useState<string | null>(null)
  const [isHeatmapExporting, setIsHeatmapExporting] = useState(false)
  const [heatmapExportError, setHeatmapExportError] = useState<string | null>(null)
  const [isHeatmapPlotLoading, setIsHeatmapPlotLoading] = useState(false)
  const [heatmapPlotError, setHeatmapPlotError] = useState<string | null>(null)
  const [heatmapPlotUrl, setHeatmapPlotUrl] = useState<string | null>(null)
  const [isHeatmapRelLoading, setIsHeatmapRelLoading] = useState(false)
  const [heatmapRelError, setHeatmapRelError] = useState<string | null>(null)
  const [heatmapRelUrl, setHeatmapRelUrl] = useState<string | null>(null)
  const [isHuSeparationLoading, setIsHuSeparationLoading] = useState(false)
  const [huSeparationError, setHuSeparationError] = useState<string | null>(null)
  const [huSeparationUrl, setHuSeparationUrl] = useState<string | null>(null)
  const [isMap256Loading, setIsMap256Loading] = useState(false)
  const [map256Error, setMap256Error] = useState<string | null>(null)
  const [map256PlotUrl, setMap256PlotUrl] = useState<string | null>(null)
  const [hasCalculatedMap256, setHasCalculatedMap256] = useState(false)
  const [contoursPlotUrl, setContoursPlotUrl] = useState<string | null>(null)
  const [contoursError, setContoursError] = useState<string | null>(null)
  const [isContoursLoading, setIsContoursLoading] = useState(false)
  const [isContoursExporting, setIsContoursExporting] = useState(false)
  const [contoursExportError, setContoursExportError] = useState<string | null>(null)
  const [hasCalculatedContours, setHasCalculatedContours] = useState(false)
  const activeUrlsRef = useRef<Set<string>>(new Set())
  const lengthPlotUrlRef = useRef<string | null>(null)
  const areaPlotUrlRef = useRef<string | null>(null)
  const medianPlotUrlRef = useRef<string | null>(null)
  const entropyPlotUrlRef = useRef<string | null>(null)
  const heatmapPlotUrlRef = useRef<string | null>(null)
  const heatmapRelUrlRef = useRef<string | null>(null)
  const huSeparationUrlRef = useRef<string | null>(null)
  const map256PlotUrlRef = useRef<string | null>(null)
  const contoursPlotUrlRef = useRef<string | null>(null)

  useEffect(() => {
    if (!dbName) {
      setError('Database is required.')
      setCells([])
      setIsLoading(false)
      return
    }

    let isActive = true
    const controller = new AbortController()

    const loadAnnotationZip = async () => {
      setIsLoading(true)
      setError(null)
      setSelectedLabel('1')
      setCells([])
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          image_type: selectedChannel,
          downscale: String(previewDownscale),
        })
        const res = await fetch(`${apiBase}/get-annotation-zip?${params.toString()}`, {
          headers: { accept: 'application/zip' },
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const buffer = await res.arrayBuffer()
        const zip = unzipSync(new Uint8Array(buffer))
        const manifestBytes = zip['manifest.json']
        if (!manifestBytes) {
          throw new Error('Manifest missing from zip')
        }
        const manifest = JSON.parse(strFromU8(manifestBytes)) as unknown
        const entries = parseManifestEntries(manifest)
        const nextCells: BulkCell[] = []

        for (const entry of entries) {
          const cellId = typeof entry.cell_id === 'string' ? entry.cell_id : ''
          const label = typeof entry.label === 'string' ? entry.label : ''
          const file = typeof entry.file === 'string' ? entry.file : ''
          if (!cellId || !file) continue
          const fileBytes = zip[file]
          if (!fileBytes) continue
          const blob = new Blob([fileBytes], { type: 'image/png' })
          const url = URL.createObjectURL(blob)
          nextCells.push({ cellId, label: normalizeLabel(label), url })
        }

        if (isActive) {
          setCells(nextCells)
        } else {
          nextCells.forEach((cell) => URL.revokeObjectURL(cell.url))
        }
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : 'Failed to load annotation zip')
          setCells([])
        }
      } finally {
        if (isActive) setIsLoading(false)
      }
    }

    void loadAnnotationZip()
    return () => {
      isActive = false
      controller.abort()
    }
  }, [apiBase, dbName, selectedChannel])

  useEffect(() => {
    const nextUrls = new Set(cells.map((cell) => cell.url))
    activeUrlsRef.current.forEach((url) => {
      if (!nextUrls.has(url)) {
        URL.revokeObjectURL(url)
      }
    })
    activeUrlsRef.current = nextUrls
  }, [cells])

  useEffect(() => {
    return () => {
      activeUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
      activeUrlsRef.current.clear()
    }
  }, [])

  const labelOptions = useMemo(() => {
    const uniqueLabels = new Set([
      'N/A',
      '1',
      '2',
      '3',
      ...cells.map((cell) => cell.label),
    ])
    return ['All', ...sortLabels([...uniqueLabels])]
  }, [cells])

  const channelOptions = [
    { value: 'ph', label: 'ph' },
    { value: 'fluo1', label: 'fluo1' },
    { value: 'fluo2', label: 'fluo2' },
  ]
  const heatmapChannelOptions = channelOptions.filter(
    (option) => option.value !== 'ph',
  )

  const analysisLabelOptions = useMemo(() => {
    const baseLabels = labelOptions.filter((option) => option !== 'All')
    const unique = new Set(baseLabels)
    unique.add('1')
    const sorted = sortLabels([...unique])
    return ['All', ...sorted]
  }, [labelOptions])

  useEffect(() => {
    if (!labelOptions.includes(selectedLabel)) {
      setSelectedLabel('1')
    }
  }, [labelOptions, selectedLabel])

  useEffect(() => {
    if (!analysisLabelOptions.includes(analysisLabel)) {
      setAnalysisLabel('1')
    }
  }, [analysisLabelOptions, analysisLabel])

  useEffect(() => {
    if (
      (analysisMode === 'heatmap' || analysisMode === 'map256') &&
      analysisChannel === 'ph'
    ) {
      setAnalysisChannel('fluo1')
    }
  }, [analysisMode, analysisChannel])

  const escapeCsvValue = (value: string) => {
    if (/[",\n]/.test(value)) {
      return `"${value.replace(/"/g, '""')}"`
    }
    return value
  }

  const filteredCells = useMemo(() => {
    if (selectedLabel === 'All') return cells
    return cells.filter((cell) => cell.label === selectedLabel)
  }, [cells, selectedLabel])

  useEffect(() => {
    if (lengthPlotUrlRef.current) {
      URL.revokeObjectURL(lengthPlotUrlRef.current)
      lengthPlotUrlRef.current = null
    }
    if (areaPlotUrlRef.current) {
      URL.revokeObjectURL(areaPlotUrlRef.current)
      areaPlotUrlRef.current = null
    }
    if (medianPlotUrlRef.current) {
      URL.revokeObjectURL(medianPlotUrlRef.current)
      medianPlotUrlRef.current = null
    }
    if (entropyPlotUrlRef.current) {
      URL.revokeObjectURL(entropyPlotUrlRef.current)
      entropyPlotUrlRef.current = null
    }
    if (heatmapPlotUrlRef.current) {
      URL.revokeObjectURL(heatmapPlotUrlRef.current)
      heatmapPlotUrlRef.current = null
    }
    if (heatmapRelUrlRef.current) {
      URL.revokeObjectURL(heatmapRelUrlRef.current)
      heatmapRelUrlRef.current = null
    }
    if (huSeparationUrlRef.current) {
      URL.revokeObjectURL(huSeparationUrlRef.current)
      huSeparationUrlRef.current = null
    }
    if (map256PlotUrlRef.current) {
      URL.revokeObjectURL(map256PlotUrlRef.current)
      map256PlotUrlRef.current = null
    }
    if (contoursPlotUrlRef.current) {
      URL.revokeObjectURL(contoursPlotUrlRef.current)
      contoursPlotUrlRef.current = null
    }
    setLengthPlotUrl(null)
    setLengthError(null)
    setLengthExportError(null)
    setAreaPlotUrl(null)
    setAreaError(null)
    setAreaExportError(null)
    setHasCalculatedLength(false)
    setHasCalculatedArea(false)
    setMedianPlotUrl(null)
    setMedianError(null)
    setMedianExportError(null)
    setHasCalculatedMedian(false)
    setEntropyPlotUrl(null)
    setEntropyError(null)
    setEntropyExportError(null)
    setHasCalculatedEntropy(false)
    setRawExportError(null)
    setHeatmapExportError(null)
    setHeatmapPlotUrl(null)
    setHeatmapPlotError(null)
    setIsHeatmapPlotLoading(false)
    setHeatmapRelUrl(null)
    setHeatmapRelError(null)
    setIsHeatmapRelLoading(false)
    setHuSeparationUrl(null)
    setHuSeparationError(null)
    setIsHuSeparationLoading(false)
    setMap256PlotUrl(null)
    setMap256Error(null)
    setIsMap256Loading(false)
    setHasCalculatedMap256(false)
    setContoursPlotUrl(null)
    setContoursError(null)
    setIsContoursLoading(false)
    setContoursExportError(null)
    setIsContoursExporting(false)
    setHasCalculatedContours(false)
  }, [analysisMode, analysisLabel, analysisChannel, dbName])

  useEffect(() => {
    return () => {
      if (lengthPlotUrlRef.current) {
        URL.revokeObjectURL(lengthPlotUrlRef.current)
        lengthPlotUrlRef.current = null
      }
      if (areaPlotUrlRef.current) {
        URL.revokeObjectURL(areaPlotUrlRef.current)
        areaPlotUrlRef.current = null
      }
      if (medianPlotUrlRef.current) {
        URL.revokeObjectURL(medianPlotUrlRef.current)
        medianPlotUrlRef.current = null
      }
      if (entropyPlotUrlRef.current) {
        URL.revokeObjectURL(entropyPlotUrlRef.current)
        entropyPlotUrlRef.current = null
      }
      if (heatmapPlotUrlRef.current) {
        URL.revokeObjectURL(heatmapPlotUrlRef.current)
        heatmapPlotUrlRef.current = null
      }
      if (heatmapRelUrlRef.current) {
        URL.revokeObjectURL(heatmapRelUrlRef.current)
        heatmapRelUrlRef.current = null
      }
      if (huSeparationUrlRef.current) {
        URL.revokeObjectURL(huSeparationUrlRef.current)
        huSeparationUrlRef.current = null
      }
      if (map256PlotUrlRef.current) {
        URL.revokeObjectURL(map256PlotUrlRef.current)
        map256PlotUrlRef.current = null
      }
      if (contoursPlotUrlRef.current) {
        URL.revokeObjectURL(contoursPlotUrlRef.current)
        contoursPlotUrlRef.current = null
      }
    }
  }, [])

  const handleCalcLength = async () => {
    if (!dbName || isLengthLoading) return
    setIsLengthLoading(true)
    setLengthError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-cell-lengths-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (lengthPlotUrlRef.current) {
        URL.revokeObjectURL(lengthPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      lengthPlotUrlRef.current = url
      setLengthPlotUrl(url)
      setHasCalculatedLength(true)
    } catch (err) {
      setLengthError(err instanceof Error ? err.message : 'Failed to calculate lengths')
      setLengthPlotUrl(null)
      setHasCalculatedLength(true)
    } finally {
      setIsLengthLoading(false)
    }
  }

  const handleExportLengthCsv = async () => {
    if (!dbName || isLengthExporting) return
    setIsLengthExporting(true)
    setLengthExportError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-cell-lengths?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const payload = (await res.json()) as unknown
      if (!Array.isArray(payload)) {
        throw new Error('Invalid response format')
      }
      const rows = payload
        .filter(
          (item): item is CellLengthPair =>
            item &&
            typeof item === 'object' &&
            typeof (item as CellLengthPair).cell_id === 'string' &&
            typeof (item as CellLengthPair).length === 'number',
        )
        .map((item) => ({
          cell_id: item.cell_id,
          length: item.length,
        }))
      if (rows.length === 0) {
        throw new Error('No lengths found for this label.')
      }
      const csvHeader = `cell_id,cell_length(\u03bcm)`
      const lines = [
        csvHeader,
        ...rows.map((row) => `${escapeCsvValue(row.cell_id)},${row.length}`),
      ]
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-cell-length.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setLengthExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsLengthExporting(false)
    }
  }

  const handleCalcArea = async () => {
    if (!dbName || isAreaLoading) return
    setIsAreaLoading(true)
    setAreaError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-cell-areas-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (areaPlotUrlRef.current) {
        URL.revokeObjectURL(areaPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      areaPlotUrlRef.current = url
      setAreaPlotUrl(url)
      setHasCalculatedArea(true)
    } catch (err) {
      setAreaError(err instanceof Error ? err.message : 'Failed to calculate areas')
      setAreaPlotUrl(null)
      setHasCalculatedArea(true)
    } finally {
      setIsAreaLoading(false)
    }
  }

  const handleCalcNormalizedMedian = async () => {
    if (!dbName || isMedianLoading) return
    setIsMedianLoading(true)
    setMedianError(null)
    try {
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: analysisChannel,
      })
      const res = await fetch(`${apiBase}/get-normalized-medians-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (medianPlotUrlRef.current) {
        URL.revokeObjectURL(medianPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      medianPlotUrlRef.current = url
      setMedianPlotUrl(url)
      setHasCalculatedMedian(true)
    } catch (err) {
      setMedianError(err instanceof Error ? err.message : 'Failed to calculate median')
      setMedianPlotUrl(null)
      setHasCalculatedMedian(true)
    } finally {
      setIsMedianLoading(false)
    }
  }

  const handleExportNormalizedMedianCsv = async () => {
    if (!dbName || isMedianExporting) return
    setIsMedianExporting(true)
    setMedianExportError(null)
    try {
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: analysisChannel,
      })
      const res = await fetch(`${apiBase}/get-normalized-medians?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const payload = (await res.json()) as unknown
      if (!Array.isArray(payload)) {
        throw new Error('Invalid response format')
      }
      const rows = payload
        .filter(
          (item): item is NormalizedMedianPair =>
            item &&
            typeof item === 'object' &&
            typeof (item as NormalizedMedianPair).cell_id === 'string' &&
            typeof (item as NormalizedMedianPair).normalized_median === 'number',
        )
        .map((item) => ({
          cell_id: item.cell_id,
          normalized_median: item.normalized_median,
        }))
      if (rows.length === 0) {
        throw new Error('No values found for this label.')
      }
      const csvHeader = 'cell_id,normalized_median'
      const lines = [
        csvHeader,
        ...rows.map((row) => `${escapeCsvValue(row.cell_id)},${row.normalized_median}`),
      ]
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-normalized-median-${analysisChannel}.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setMedianExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsMedianExporting(false)
    }
  }

  const handleCalcEntropy = async () => {
    if (!dbName || isEntropyLoading) return
    setIsEntropyLoading(true)
    setEntropyError(null)
    try {
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: analysisChannel,
      })
      const res = await fetch(`${apiBase}/get-entropy-metrics-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (entropyPlotUrlRef.current) {
        URL.revokeObjectURL(entropyPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      entropyPlotUrlRef.current = url
      setEntropyPlotUrl(url)
      setHasCalculatedEntropy(true)
    } catch (err) {
      setEntropyError(err instanceof Error ? err.message : 'Failed to calculate entropy')
      setEntropyPlotUrl(null)
      setHasCalculatedEntropy(true)
    } finally {
      setIsEntropyLoading(false)
    }
  }

  const handleExportEntropyCsv = async () => {
    if (!dbName || isEntropyExporting) return
    setIsEntropyExporting(true)
    setEntropyExportError(null)
    try {
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: analysisChannel,
      })
      const res = await fetch(`${apiBase}/get-entropy-metrics?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const payload = (await res.json()) as unknown
      if (!Array.isArray(payload)) {
        throw new Error('Invalid response format')
      }
      const rows = payload
        .filter(
          (item): item is EntropyMetricsPair =>
            item &&
            typeof item === 'object' &&
            typeof (item as EntropyMetricsPair).cell_id === 'string' &&
            typeof (item as EntropyMetricsPair).entropy_norm === 'number' &&
            typeof (item as EntropyMetricsPair).sparsity === 'number',
        )
        .map((item) => ({
          cell_id: item.cell_id,
          entropy_norm: item.entropy_norm,
          sparsity: item.sparsity,
        }))
      if (rows.length === 0) {
        throw new Error('No values found for this label.')
      }
      const csvHeader = 'cell_id,entropy_norm,sparsity'
      const lines = [
        csvHeader,
        ...rows.map(
          (row) =>
            `${escapeCsvValue(row.cell_id)},${row.entropy_norm},${row.sparsity}`,
        ),
      ]
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-entropy-${analysisChannel}.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setEntropyExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsEntropyExporting(false)
    }
  }

  const handleExportRawData = async () => {
    if (!dbName || isRawExporting) return
    setIsRawExporting(true)
    setRawExportError(null)
    try {
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: analysisChannel,
      })
      const res = await fetch(`${apiBase}/get-raw-intensities?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const payload = (await res.json()) as unknown
      if (!Array.isArray(payload)) {
        throw new Error('Invalid response format')
      }
      const rows = payload
        .filter(
          (item): item is RawIntensityPair =>
            item &&
            typeof item === 'object' &&
            typeof (item as RawIntensityPair).cell_id === 'string' &&
            Array.isArray((item as RawIntensityPair).intensities),
        )
        .map((item) => ({
          cell_id: item.cell_id,
          intensities: item.intensities.filter(
            (value) => typeof value === 'number' && Number.isFinite(value),
          ),
        }))
      if (rows.length === 0) {
        throw new Error('No raw intensities found for this label.')
      }
      const lines = rows.map((row) =>
        [escapeCsvValue(row.cell_id), ...row.intensities.map((value) => String(value))].join(
          ',',
        ),
      )
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-raw-${analysisChannel}.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setRawExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsRawExporting(false)
    }
  }

  const handleExportHeatmapCsv = async () => {
    if (!dbName || isHeatmapExporting) return
    setIsHeatmapExporting(true)
    setHeatmapExportError(null)
    try {
      const heatmapChannel = analysisChannel === 'ph' ? 'fluo1' : analysisChannel
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: heatmapChannel,
      })
      const res = await fetch(`${apiBase}/get-heatmap-vectors-csv?${params.toString()}`, {
        headers: { accept: 'text/csv' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No heatmap vectors found for this label.')
      }
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-heatmap-${heatmapChannel}.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setHeatmapExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsHeatmapExporting(false)
    }
  }

  const handleGenerateHeatmapPlot = async () => {
    if (!dbName || isHeatmapPlotLoading) return
    setIsHeatmapPlotLoading(true)
    setHeatmapPlotError(null)
    try {
      const heatmapChannel = analysisChannel === 'ph' ? 'fluo1' : analysisChannel
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: heatmapChannel,
      })
      const res = await fetch(`${apiBase}/get-heatmap-abs-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No heatmap vectors found for this label.')
      }
      if (heatmapPlotUrlRef.current) {
        URL.revokeObjectURL(heatmapPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      heatmapPlotUrlRef.current = url
      setHeatmapPlotUrl(url)
    } catch (err) {
      setHeatmapPlotError(
        err instanceof Error ? err.message : 'Failed to generate heatmap plot',
      )
      setHeatmapPlotUrl(null)
    } finally {
      setIsHeatmapPlotLoading(false)
    }
  }

  const handleGenerateHeatmapRelPlot = async () => {
    if (!dbName || isHeatmapRelLoading) return
    setIsHeatmapRelLoading(true)
    setHeatmapRelError(null)
    try {
      const heatmapChannel = analysisChannel === 'ph' ? 'fluo1' : analysisChannel
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: heatmapChannel,
      })
      const res = await fetch(`${apiBase}/get-heatmap-rel-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No heatmap vectors found for this label.')
      }
      if (heatmapRelUrlRef.current) {
        URL.revokeObjectURL(heatmapRelUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      heatmapRelUrlRef.current = url
      setHeatmapRelUrl(url)
    } catch (err) {
      setHeatmapRelError(
        err instanceof Error ? err.message : 'Failed to generate heatmap (rel) plot',
      )
      setHeatmapRelUrl(null)
    } finally {
      setIsHeatmapRelLoading(false)
    }
  }

  const handleGenerateHuSeparation = async () => {
    if (!dbName || isHuSeparationLoading) return
    setIsHuSeparationLoading(true)
    setHuSeparationError(null)
    try {
      const heatmapChannel = analysisChannel === 'ph' ? 'fluo1' : analysisChannel
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: heatmapChannel,
      })
      const res = await fetch(`${apiBase}/get-hu-separation-overlay?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No heatmap vectors found for this label.')
      }
      if (huSeparationUrlRef.current) {
        URL.revokeObjectURL(huSeparationUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      huSeparationUrlRef.current = url
      setHuSeparationUrl(url)
    } catch (err) {
      setHuSeparationError(
        err instanceof Error ? err.message : 'Failed to generate HU separation overlay',
      )
      setHuSeparationUrl(null)
    } finally {
      setIsHuSeparationLoading(false)
    }
  }

  const handleGenerateContoursGrid = async () => {
    if (!dbName || isContoursLoading) return
    setIsContoursLoading(true)
    setContoursError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-contours-grid-plot?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (contoursPlotUrlRef.current) {
        URL.revokeObjectURL(contoursPlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      contoursPlotUrlRef.current = url
      setContoursPlotUrl(url)
      setHasCalculatedContours(true)
    } catch (err) {
      setContoursError(
        err instanceof Error ? err.message : 'Failed to generate contours grid',
      )
      setContoursPlotUrl(null)
      setHasCalculatedContours(true)
    } finally {
      setIsContoursLoading(false)
    }
  }

  const handleExportContoursCsv = async () => {
    if (!dbName || isContoursExporting) return
    setIsContoursExporting(true)
    setContoursExportError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-contours-grid-csv?${params.toString()}`, {
        headers: { accept: 'text/csv' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No contours found for this label.')
      }
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-contours.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setContoursExportError(
        err instanceof Error ? err.message : 'Failed to export contours CSV',
      )
    } finally {
      setIsContoursExporting(false)
    }
  }

  const handleGenerateMap256 = async () => {
    if (!dbName || isMap256Loading) return
    setIsMap256Loading(true)
    setMap256Error(null)
    try {
      const mapChannel = analysisChannel === 'ph' ? 'fluo1' : analysisChannel
      const params = new URLSearchParams({
        dbname: dbName,
        label: analysisLabel,
        channel: mapChannel,
      })
      const res = await fetch(`${apiBase}/get-map256-strip?${params.toString()}`, {
        headers: { accept: 'image/png' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const blob = await res.blob()
      if (blob.size === 0) {
        throw new Error('No map256 data found for this label.')
      }
      if (map256PlotUrlRef.current) {
        URL.revokeObjectURL(map256PlotUrlRef.current)
      }
      const url = URL.createObjectURL(blob)
      map256PlotUrlRef.current = url
      setMap256PlotUrl(url)
      setHasCalculatedMap256(true)
    } catch (err) {
      setMap256Error(err instanceof Error ? err.message : 'Failed to generate map256')
      setMap256PlotUrl(null)
      setHasCalculatedMap256(true)
    } finally {
      setIsMap256Loading(false)
    }
  }

  const handleExportAreaCsv = async () => {
    if (!dbName || isAreaExporting) return
    setIsAreaExporting(true)
    setAreaExportError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName, label: analysisLabel })
      const res = await fetch(`${apiBase}/get-cell-areas?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const payload = (await res.json()) as unknown
      if (!Array.isArray(payload)) {
        throw new Error('Invalid response format')
      }
      const rows = payload
        .filter(
          (item): item is CellAreaPair =>
            item &&
            typeof item === 'object' &&
            typeof (item as CellAreaPair).cell_id === 'string' &&
            typeof (item as CellAreaPair).area === 'number',
        )
        .map((item) => ({
          cell_id: item.cell_id,
          area: item.area,
        }))
      if (rows.length === 0) {
        throw new Error('No areas found for this label.')
      }
      const csvHeader = 'cell_id,cell_area(px^2)'
      const lines = [
        csvHeader,
        ...rows.map((row) => `${escapeCsvValue(row.cell_id)},${row.area}`),
      ]
      const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' })
      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        analysisLabel === 'All'
          ? 'all'
          : analysisLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-cell-area.csv`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setAreaExportError(err instanceof Error ? err.message : 'Failed to export CSV')
    } finally {
      setIsAreaExporting(false)
    }
  }

  const handleExport = async () => {
    if (isExporting || isLoading || filteredCells.length === 0) return
    setIsExporting(true)
    setExportError(null)
    const cleanupUrls: string[] = []
    try {
      let exportCells = filteredCells.map((cell) => ({ cellId: cell.cellId, url: cell.url }))
      if (selectedChannel === 'ph') {
        const params = new URLSearchParams({
          dbname: dbName,
          image_type: selectedChannel,
          raw: 'true',
        })
        const res = await fetch(`${apiBase}/get-annotation-zip?${params.toString()}`, {
          headers: { accept: 'application/zip' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const buffer = await res.arrayBuffer()
        const zip = unzipSync(new Uint8Array(buffer))
        const manifestBytes = zip['manifest.json']
        if (!manifestBytes) {
          throw new Error('Manifest missing from zip')
        }
        const manifest = JSON.parse(strFromU8(manifestBytes)) as unknown
        const entries = parseManifestEntries(manifest)
        const allowedIds = new Set(filteredCells.map((cell) => cell.cellId))
        const fileMap = new Map<string, Uint8Array>()
        for (const entry of entries) {
          const cellId = typeof entry.cell_id === 'string' ? entry.cell_id : ''
          const file = typeof entry.file === 'string' ? entry.file : ''
          if (!cellId || !file || !allowedIds.has(cellId)) continue
          const fileBytes = zip[file]
          if (!fileBytes) continue
          fileMap.set(cellId, fileBytes)
        }
        exportCells = filteredCells
          .map((cell) => {
            const fileBytes = fileMap.get(cell.cellId)
            if (!fileBytes) return null
            const blob = new Blob([fileBytes], { type: 'image/png' })
            const url = URL.createObjectURL(blob)
            cleanupUrls.push(url)
            return { cellId: cell.cellId, url }
          })
          .filter((cell): cell is { cellId: string; url: string } => Boolean(cell))
      }
      if (exportCells.length === 0) {
        throw new Error('No images found for export')
      }

      const imagePromises = exportCells.map(
        (cell) =>
          new Promise<HTMLImageElement>((resolve, reject) => {
            const img = new Image()
            img.decoding = 'async'
            img.onload = () => resolve(img)
            img.onerror = () => reject(new Error(`Failed to load ${cell.cellId}`))
            img.src = cell.url
          }),
      )
      const images = await Promise.all(imagePromises)
      const cellSize = Math.max(
        ...images.map((img) => Math.max(img.naturalWidth || img.width, img.naturalHeight || img.height)),
      )
      const columns = Math.ceil(Math.sqrt(images.length))
      const rows = Math.ceil(images.length / columns)
      const canvas = document.createElement('canvas')
      canvas.width = columns * cellSize
      canvas.height = rows * cellSize
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Canvas is not supported in this browser')
      }
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      images.forEach((img, index) => {
        const col = index % columns
        const row = Math.floor(index / columns)
        const width = img.naturalWidth || img.width
        const height = img.naturalHeight || img.height
        const offsetX = col * cellSize + Math.floor((cellSize - width) / 2)
        const offsetY = row * cellSize + Math.floor((cellSize - height) / 2)
        ctx.drawImage(img, offsetX, offsetY)
      })

      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((nextBlob) => {
          if (!nextBlob) {
            reject(new Error('Failed to export image'))
            return
          }
          resolve(nextBlob)
        }, 'image/png')
      })

      const safeDb = (dbName || 'db').replace(/\.db$/i, '').replace(/[^a-zA-Z0-9_-]/g, '_')
      const safeLabel =
        selectedLabel === 'All'
          ? 'all'
          : selectedLabel.replace(/[^a-zA-Z0-9_-]/g, '_')
      const filename = `bulk-${safeDb}-${safeLabel}-${selectedChannel}.png`
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      cleanupUrls.forEach((url) => URL.revokeObjectURL(url))
      setIsExporting(false)
    }
  }

  const gridColumns = {
    base: 'repeat(2, minmax(0, 1fr))',
    md: 'repeat(4, minmax(0, 1fr))',
    lg: 'repeat(7, minmax(0, 1fr))',
  }

  return (
    <Box
      minH={{ base: '100vh', lg: scaledViewportHeight }}
      h={{ base: 'auto', lg: scaledViewportHeight }}
      bg="sand.50"
      color="ink.900"
      display="flex"
      flexDirection="column"
      overflow={{ base: 'visible', lg: 'hidden' }}
      style={{ zoom: bulkZoom }}
    >
      <AppHeader>
        <HStack spacing="3">
          <Box
            as="img"
            src="/favicon.png"
            alt="PhenoPixel logo"
            w="1.25rem"
            h="1.25rem"
            objectFit="contain"
          />
          <Heading size="md" letterSpacing="0.08em">
            PhenoPixel 6.0
          </Heading>
          <Badge
            bg="sand.100"
            color="ink.700"
            borderRadius="full"
            px="2"
            py="1"
            fontSize="0.6rem"
            letterSpacing="0.2em"
            textTransform="uppercase"
          >
            Bulk Engine
          </Badge>
        </HStack>
        <HStack spacing="4" align="center">
          <BreadcrumbRoot fontSize="sm" color="ink.700">
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink as={RouterLink} to="/">
                  Dashboard
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator>/</BreadcrumbSeparator>
              <BreadcrumbItem>
                <BreadcrumbLink as={RouterLink} to="/databases">
                  Databases
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator>/</BreadcrumbSeparator>
              <BreadcrumbItem>
                <BreadcrumbCurrentLink color="ink.900">Bulk Engine</BreadcrumbCurrentLink>
              </BreadcrumbItem>
            </BreadcrumbList>
          </BreadcrumbRoot>
          <ReloadButton />
          <ThemeToggle />
        </HStack>
      </AppHeader>

      <Container
        maxW="96rem"
        py={{ base: 6, md: 8, lg: 4 }}
        flex="1"
        display="flex"
        flexDirection="column"
        minH="0"
      >
        <Stack spacing={{ base: 5, lg: 4 }} flex="1" minH="0">
          <HStack justify="space-between" flexWrap="wrap" gap="3">
            <Text fontSize="sm" color="ink.700">
              Database: {dbName || 'Not selected'}
            </Text>
            <Text fontSize="xs" color="ink.700">
              Showing {filteredCells.length} / {cells.length}
            </Text>
          </HStack>

          {isLoading && (
            <HStack spacing="3" color="ink.700">
              <Spinner size="sm" />
              <Text fontSize="sm">Generating images...</Text>
            </HStack>
          )}

          {!isLoading && error && (
            <Text fontSize="sm" color="violet.300">
              {error}
            </Text>
          )}

          {!isLoading && !error && cells.length === 0 && (
            <Text fontSize="sm" color="ink.700">
              No annotation images found.
            </Text>
          )}

          {!isLoading && !error && cells.length > 0 && (
            <Grid
              templateColumns={{ base: 'minmax(0, 1fr)', lg: 'repeat(2, minmax(0, 1fr))' }}
              templateRows={{ base: 'auto', lg: 'minmax(0, 1fr)' }}
              gap="6"
              alignItems="start"
              flex="1"
              minH="0"
              h={{ base: 'auto', lg: '100%' }}
            >
              <Box
                bg="sand.100"
                border="1px solid"
                borderColor="sand.200"
                borderRadius="xl"
                p={{ base: 3, md: 4 }}
                display="flex"
                flexDirection="column"
                minH="0"
                h={{ base: 'auto', lg: '100%' }}
              >
                <HStack justify="space-between" flexWrap="wrap" gap="3" mb="4">
                  <HStack spacing="2">
                    <Badge
                      bg="sand.200"
                      color="ink.700"
                      borderRadius="full"
                      px="2"
                      py="1"
                      fontSize="0.6rem"
                      letterSpacing="0.18em"
                      textTransform="uppercase"
                    >
                      Cells
                    </Badge>
                    <Text fontSize="xs" color="ink.700">
                      Preview panel
                    </Text>
                    <Text fontSize="xs" color="ink.700">
                      Filter by manual label
                    </Text>
                  </HStack>
                  <HStack spacing="4" align="flex-start" flexWrap="wrap">
                    <Box minW="12rem">
                      <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                        Manual label
                      </Text>
                      <Stack spacing="1">
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={selectedLabel}
                            onChange={(event) => setSelectedLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            {labelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                        <Text fontSize="xs" color="ink.700">
                          Cells: {filteredCells.length}
                        </Text>
                      </Stack>
                    </Box>
                    <Box minW="10rem">
                      <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                        Channel
                      </Text>
                      <Stack spacing="1">
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={selectedChannel}
                            onChange={(event) => setSelectedChannel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            {channelOptions.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                        <Text fontSize="xs" color="ink.700">
                          Channel: {selectedChannel}
                        </Text>
                      </Stack>
                    </Box>
                    <Box>
                      <Text
                        fontSize="xs"
                        letterSpacing="0.18em"
                        color="ink.700"
                        mb="1"
                        visibility="hidden"
                      >
                        Export
                      </Text>
                      <Button
                        size="sm"
                        h="2.25rem"
                        bg="tide.500"
                        color="ink.900"
                        _hover={{ bg: 'tide.400' }}
                        onClick={handleExport}
                        isDisabled={filteredCells.length === 0 || isLoading || isExporting}
                        opacity={filteredCells.length === 0 ? 0.5 : 1}
                      >
                        {isExporting ? 'Exporting...' : 'Export'}
                      </Button>
                    </Box>
                  </HStack>
                </HStack>
                {exportError && (
                  <Text fontSize="xs" color="violet.300" mb="3">
                    {exportError}
                  </Text>
                )}

                <Box
                  flex="1"
                  minH="0"
                  overflowY={{ base: 'visible', lg: 'auto' }}
                  maxH="100%"
                  pr={{ base: 0, lg: 1 }}
                >
                  {filteredCells.length === 0 ? (
                    <Text fontSize="sm" color="ink.700">
                      No cells match this label.
                    </Text>
                  ) : (
                    <Grid templateColumns={gridColumns} gap="2" pb="1">
                      {filteredCells.map((cell) => (
                        <Box
                          key={cell.cellId}
                          bg="sand.50"
                          borderRadius="md"
                          p="1"
                          borderWidth="1px"
                          borderColor="sand.200"
                        >
                          <AspectRatio ratio={1}>
                            <Box
                              as="img"
                              src={cell.url}
                              alt={`Cell ${cell.cellId}`}
                              objectFit="cover"
                              borderRadius="sm"
                            />
                          </AspectRatio>
                          <Text fontSize="0.6rem" mt="1" color="ink.700" textAlign="center">
                            {cell.cellId}
                          </Text>
                        </Box>
                      ))}
                    </Grid>
                  )}
                </Box>
              </Box>

              <Box
                bg="sand.100"
                border="1px solid"
                borderColor="sand.200"
                borderRadius="xl"
                p={{ base: 3, md: 4 }}
                minH={{ base: '240px', lg: '0' }}
                display="flex"
                flexDirection="column"
                h={{ base: 'auto', lg: '100%' }}
              >
                <Badge
                  bg="sand.200"
                  color="ink.700"
                  borderRadius="full"
                  px="2"
                  py="1"
                  fontSize="0.6rem"
                  letterSpacing="0.18em"
                  textTransform="uppercase"
                >
                  Bulk Engine
                </Badge>
                <Box
                  flex="1"
                  minH="0"
                  overflowY={{ base: 'visible', lg: 'auto' }}
                  pr={{ base: 0, lg: 1 }}
                >
                  <Text fontSize="xs" color="ink.700" mt="3">
                    Analysis mode
                  </Text>
                  <Stack spacing="3" mt="2">
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={analysisMode}
                        onChange={(event) => setAnalysisMode(event.target.value)}
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="sm"
                        h="2.25rem"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        <option value="cell-length">Cell length</option>
                        <option value="cell-area">Cell area</option>
                        <option value="normalized-median">Normalized median</option>
                        <option value="entropy">Entropy</option>
                        <option value="heatmap">Heatmap</option>
                        <option value="contours">Contours</option>
                        <option value="map256">Map256</option>
                        <option value="raw-data">Raw data</option>
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </Stack>
                  <Text fontSize="xs" color="ink.700" mt="4">
                    Parameter settings
                  </Text>
                  {analysisMode === 'cell-length' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Cell length (um)
                      </Text>
                      <Box maxW="12rem">
                        <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                          Manual label
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={analysisLabel}
                            onChange={(event) => setAnalysisLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            {analysisLabelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                      </Box>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleCalcLength}
                          isDisabled={!dbName || isLengthLoading}
                        >
                          {isLengthLoading ? 'Calculating...' : 'Calc length'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportLengthCsv}
                          isDisabled={!dbName || isLengthExporting}
                        >
                          {isLengthExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {lengthExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {lengthExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Result
                      </Text>
                      {lengthError && (
                        <Text fontSize="xs" color="violet.300">
                          {lengthError}
                        </Text>
                      )}
                      {!lengthError && hasCalculatedLength && !lengthPlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No lengths found for this label.
                        </Text>
                      )}
                      {lengthPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                        >
                          <Box
                            as="img"
                            src={lengthPlotUrl}
                            alt="Cell length boxplot"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'cell-area' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Cell area (px^2)
                      </Text>
                      <Box maxW="12rem">
                        <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                          Manual label
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={analysisLabel}
                            onChange={(event) => setAnalysisLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            {analysisLabelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                      </Box>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleCalcArea}
                          isDisabled={!dbName || isAreaLoading}
                        >
                          {isAreaLoading ? 'Calculating...' : 'Calc area'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportAreaCsv}
                          isDisabled={!dbName || isAreaExporting}
                        >
                          {isAreaExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {areaExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {areaExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Result
                      </Text>
                      {areaError && (
                        <Text fontSize="xs" color="violet.300">
                          {areaError}
                        </Text>
                      )}
                      {!areaError && hasCalculatedArea && !areaPlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No areas found for this label.
                        </Text>
                      )}
                      {areaPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                        >
                          <Box
                            as="img"
                            src={areaPlotUrl}
                            alt="Cell area boxplot"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'normalized-median' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Normalized median
                      </Text>
                      <HStack spacing="4" align="flex-start" flexWrap="wrap">
                        <Box maxW="12rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Manual label
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisLabel}
                              onChange={(event) => setAnalysisLabel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {analysisLabelOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                        <Box maxW="10rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Channel
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisChannel}
                              onChange={(event) => setAnalysisChannel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {channelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleCalcNormalizedMedian}
                          isDisabled={!dbName || isMedianLoading}
                        >
                          {isMedianLoading ? 'Calculating...' : 'Calc median'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportNormalizedMedianCsv}
                          isDisabled={!dbName || isMedianExporting}
                        >
                          {isMedianExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {medianExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {medianExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Result
                      </Text>
                      {medianError && (
                        <Text fontSize="xs" color="violet.300">
                          {medianError}
                        </Text>
                      )}
                      {!medianError && hasCalculatedMedian && !medianPlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No values found for this label.
                        </Text>
                      )}
                      {medianPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                        >
                          <Box
                            as="img"
                            src={medianPlotUrl}
                            alt="Normalized median boxplot"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'entropy' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Entropy / (1 - sparsity)
                      </Text>
                      <HStack spacing="4" align="flex-start" flexWrap="wrap">
                        <Box maxW="12rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Manual label
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisLabel}
                              onChange={(event) => setAnalysisLabel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {analysisLabelOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                        <Box maxW="10rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Channel
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisChannel}
                              onChange={(event) => setAnalysisChannel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {channelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleCalcEntropy}
                          isDisabled={!dbName || isEntropyLoading}
                        >
                          {isEntropyLoading ? 'Calculating...' : 'Calc entropy'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportEntropyCsv}
                          isDisabled={!dbName || isEntropyExporting}
                        >
                          {isEntropyExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {entropyExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {entropyExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Result
                      </Text>
                      {entropyError && (
                        <Text fontSize="xs" color="violet.300">
                          {entropyError}
                        </Text>
                      )}
                      {!entropyError && hasCalculatedEntropy && !entropyPlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No values found for this label.
                        </Text>
                      )}
                      {entropyPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                        >
                          <Box
                            as="img"
                            src={entropyPlotUrl}
                            alt="Entropy / sparsity boxplot"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'heatmap' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Heatmap vectors
                      </Text>
                      <HStack spacing="4" align="flex-start" flexWrap="wrap">
                        <Box maxW="12rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Manual label
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisLabel}
                              onChange={(event) => setAnalysisLabel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {analysisLabelOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                        <Box maxW="10rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Channel
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisChannel}
                              onChange={(event) => setAnalysisChannel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {heatmapChannelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportHeatmapCsv}
                          isDisabled={!dbName || isHeatmapExporting}
                        >
                          {isHeatmapExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleGenerateHeatmapPlot}
                          isDisabled={!dbName || isHeatmapPlotLoading}
                        >
                          {isHeatmapPlotLoading ? 'Generating...' : 'Bulk heatmap'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleGenerateHeatmapRelPlot}
                          isDisabled={!dbName || isHeatmapRelLoading}
                        >
                          {isHeatmapRelLoading ? 'Generating...' : 'Heatmap (rel)'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleGenerateHuSeparation}
                          isDisabled={!dbName || isHuSeparationLoading}
                        >
                          {isHuSeparationLoading ? 'Generating...' : 'HU Separation'}
                        </Button>
                      </HStack>
                      {heatmapExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {heatmapExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Output
                      </Text>
                      {heatmapPlotError && (
                        <Text fontSize="xs" color="violet.300">
                          {heatmapPlotError}
                        </Text>
                      )}
                      {heatmapRelError && (
                        <Text fontSize="xs" color="violet.300">
                          {heatmapRelError}
                        </Text>
                      )}
                      {huSeparationError && (
                        <Text fontSize="xs" color="violet.300">
                          {huSeparationError}
                        </Text>
                      )}
                      {heatmapPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                          mt="2"
                        >
                          <Box
                            as="img"
                            src={heatmapPlotUrl}
                            alt="Bulk heatmap"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                      {heatmapRelUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                          mt="2"
                        >
                          <Box
                            as="img"
                            src={heatmapRelUrl}
                            alt="Bulk heatmap (rel)"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                      {huSeparationUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                          mt="2"
                        >
                          <Box
                            as="img"
                            src={huSeparationUrl}
                            alt="HU separation overlay"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                      <Text fontSize="sm" color="ink.700">
                        CSV rows are u1 and G vectors per cell for heatmap rendering.
                      </Text>
                    </Stack>
                  )}
                  {analysisMode === 'contours' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Contours grid
                      </Text>
                      <Box maxW="12rem">
                        <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                          Manual label
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={analysisLabel}
                            onChange={(event) => setAnalysisLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            {analysisLabelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                      </Box>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleGenerateContoursGrid}
                          isDisabled={!dbName || isContoursLoading}
                        >
                          {isContoursLoading ? 'Generating...' : 'Generate contours'}
                        </Button>
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportContoursCsv}
                          isDisabled={!dbName || isContoursExporting}
                        >
                          {isContoursExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {contoursExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {contoursExportError}
                        </Text>
                      )}
                      {contoursError && (
                        <Text fontSize="xs" color="violet.300">
                          {contoursError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Result
                      </Text>
                      {!contoursError && hasCalculatedContours && !contoursPlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No contours found for this label.
                        </Text>
                      )}
                      {contoursPlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                        >
                          <Box
                            as="img"
                            src={contoursPlotUrl}
                            alt="Contours grid"
                            width="100%"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'map256' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Map256 strip
                      </Text>
                      <HStack spacing="4" align="flex-start" flexWrap="wrap">
                        <Box maxW="12rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Manual label
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisLabel}
                              onChange={(event) => setAnalysisLabel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {analysisLabelOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                        <Box maxW="10rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Channel
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisChannel}
                              onChange={(event) => setAnalysisChannel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {heatmapChannelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleGenerateMap256}
                          isDisabled={!dbName || isMap256Loading}
                        >
                          {isMap256Loading ? 'Generating...' : 'Generate map256'}
                        </Button>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Output
                      </Text>
                      {map256Error && (
                        <Text fontSize="xs" color="violet.300">
                          {map256Error}
                        </Text>
                      )}
                      {!map256Error && hasCalculatedMap256 && !map256PlotUrl && (
                        <Text fontSize="sm" color="ink.700">
                          No values found for this label.
                        </Text>
                      )}
                      {map256PlotUrl && (
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="md"
                          p="2"
                          mt="2"
                          overflowX="auto"
                        >
                          <Box
                            as="img"
                            src={map256PlotUrl}
                            alt="Map256 strip"
                            display="block"
                            maxW="none"
                            height="auto"
                          />
                        </Box>
                      )}
                    </Stack>
                  )}
                  {analysisMode === 'raw-data' && (
                    <Stack spacing="3" mt="2">
                      <Text fontSize="sm" fontWeight="600" color="ink.900">
                        Raw intensity data
                      </Text>
                      <HStack spacing="4" align="flex-start" flexWrap="wrap">
                        <Box maxW="12rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Manual label
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisLabel}
                              onChange={(event) => setAnalysisLabel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {analysisLabelOptions.map((option) => (
                                <option key={option} value={option}>
                                  {option}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                        <Box maxW="10rem">
                          <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                            Channel
                          </Text>
                          <NativeSelect.Root>
                            <NativeSelect.Field
                              value={analysisChannel}
                              onChange={(event) => setAnalysisChannel(event.target.value)}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h="2.25rem"
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            >
                              {channelOptions.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </NativeSelect.Field>
                            <NativeSelect.Indicator color="ink.700" />
                          </NativeSelect.Root>
                        </Box>
                      </HStack>
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Execution
                      </Text>
                      <HStack spacing="3" flexWrap="wrap">
                        <Button
                          size="sm"
                          bg="tide.500"
                          color="ink.900"
                          _hover={{ bg: 'tide.400' }}
                          onClick={handleExportRawData}
                          isDisabled={!dbName || isRawExporting}
                        >
                          {isRawExporting ? 'Exporting...' : 'Export CSV'}
                        </Button>
                      </HStack>
                      {rawExportError && (
                        <Text fontSize="xs" color="violet.300">
                          {rawExportError}
                        </Text>
                      )}
                      <Text fontSize="xs" color="ink.700" mt="3">
                        Output
                      </Text>
                      <Text fontSize="sm" color="ink.700">
                        One row per cell: cell_id followed by raw intensities inside the contour.
                      </Text>
                    </Stack>
                  )}
                </Box>
              </Box>
            </Grid>
          )}
        </Stack>
      </Container>
    </Box>
  )
}
