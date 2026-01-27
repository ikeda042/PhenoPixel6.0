import { useCallback, useEffect, useMemo, useState } from 'react'
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
  Checkbox,
  Container,
  Grid,
  Heading,
  HStack,
  Input,
  NativeSelect,
  Slider,
  Spinner,
  Stack,
  Text,
} from '@chakra-ui/react'
import { ArrowLeft, ArrowRight } from 'lucide-react'
import AppHeader from '../components/AppHeader'
import ReloadButton from '../components/ReloadButton'
import ThemeToggle from '../components/ThemeToggle'
import { getApiBase } from '../utils/apiBase'

type ChannelKey = 'ph' | 'fluo1' | 'fluo2'
type ReplotChannel = ChannelKey | 'overlay'
type OverlayOptions = {
  contour: boolean
  scale: boolean
}

const channels: { key: ChannelKey; label: string }[] = [
  { key: 'ph', label: 'PH' },
  { key: 'fluo1', label: 'FLUO1' },
  { key: 'fluo2', label: 'FLUO2' },
]

const channelOptions: { value: ChannelKey; label: string }[] = [
  { value: 'ph', label: 'PH' },
  { value: 'fluo1', label: 'Fluo1' },
  { value: 'fluo2', label: 'Fluo2' },
]
const replotChannelOptions: { value: ReplotChannel; label: string }[] = [
  ...channelOptions,
  { value: 'overlay', label: 'Overlay' },
]

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

const normalizeManualLabel = (label: string | null) => {
  const trimmed = (label ?? '').trim()
  return trimmed ? trimmed : null
}

const parseCellId = (cellId: string) => {
  const match = /^F(\d+)C(\d+)$/i.exec(cellId)
  if (!match) return null
  return { frame: Number(match[1]), cell: Number(match[2]) }
}

const sortCellIds = (ids: string[]) =>
  [...ids].sort((a, b) => {
    const parsedA = parseCellId(a)
    const parsedB = parseCellId(b)
    if (parsedA && parsedB) {
      if (parsedA.frame !== parsedB.frame) {
        return parsedA.frame - parsedB.frame
      }
      return parsedA.cell - parsedB.cell
    }
    if (parsedA) return -1
    if (parsedB) return 1
    return a.localeCompare(b)
  })

export default function CellsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const dbName = searchParams.get('db') ?? ''
  const requestedCellId = searchParams.get('cell_id') ?? searchParams.get('cell') ?? ''
  const apiBase = useMemo(() => getApiBase(), [])

  const [cellIds, setCellIds] = useState<string[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [manualLabel, setManualLabel] = useState<string | null>(null)
  const [isLoadingLabel, setIsLoadingLabel] = useState(false)
  const [isUpdatingLabel, setIsUpdatingLabel] = useState(false)
  const [manualLabelError, setManualLabelError] = useState<string | null>(null)
  const [labelOptions, setLabelOptions] = useState<string[]>(['All'])
  const [selectedLabel, setSelectedLabel] = useState('All')
  const [overlayOptions, setOverlayOptions] = useState<OverlayOptions>({
    contour: true,
    scale: true,
  })
  const [contourMode, setContourMode] = useState<
    | 'contour'
    | 'replot'
    | 'overlay'
    | 'overlay-fluo'
    | 'heatmap'
    | 'distribution'
    | 'map256'
  >('overlay')
  const [contourData, setContourData] = useState<number[][]>([])
  const [isLoadingContour, setIsLoadingContour] = useState(false)
  const [contourError, setContourError] = useState<string | null>(null)
  const [replotUrl, setReplotUrl] = useState<string | null>(null)
  const [isLoadingReplot, setIsLoadingReplot] = useState(false)
  const [replotError, setReplotError] = useState<string | null>(null)
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null)
  const [isLoadingOverlay, setIsLoadingOverlay] = useState(false)
  const [overlayError, setOverlayError] = useState<string | null>(null)
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null)
  const [isLoadingHeatmap, setIsLoadingHeatmap] = useState(false)
  const [heatmapError, setHeatmapError] = useState<string | null>(null)
  const [heatmapChannel, setHeatmapChannel] = useState<'fluo1' | 'fluo2'>('fluo1')
  const [map256Url, setMap256Url] = useState<string | null>(null)
  const [isLoadingMap256, setIsLoadingMap256] = useState(false)
  const [map256Error, setMap256Error] = useState<string | null>(null)
  const [map256Channel, setMap256Channel] = useState<'fluo1' | 'fluo2'>('fluo1')
  const [map256JetUrl, setMap256JetUrl] = useState<string | null>(null)
  const [map256JetError, setMap256JetError] = useState<string | null>(null)
  const [distributionUrl, setDistributionUrl] = useState<string | null>(null)
  const [isLoadingDistribution, setIsLoadingDistribution] = useState(false)
  const [distributionError, setDistributionError] = useState<string | null>(null)
  const [distributionChannel, setDistributionChannel] = useState<ChannelKey>('fluo1')
  const [replotChannel, setReplotChannel] = useState<ReplotChannel>('fluo1')
  const [contourRefreshKey, setContourRefreshKey] = useState(0)
  const [modificationMode, setModificationMode] = useState<'elastic' | 'optical-boost'>(
    'elastic',
  )
  const [elasticDelta, setElasticDelta] = useState(0)
  const [isApplyingModification, setIsApplyingModification] = useState(false)
  const [modificationError, setModificationError] = useState<string | null>(null)
  const [imageDimensions, setImageDimensions] = useState<{
    width: number
    height: number
  } | null>(null)
  const [images, setImages] = useState<Record<ChannelKey, string | null>>({
    ph: null,
    fluo1: null,
    fluo2: null,
  })
  const [missingChannels, setMissingChannels] = useState<Record<ChannelKey, boolean>>({
    ph: false,
    fluo1: false,
    fluo2: false,
  })
  const [isLoadingIds, setIsLoadingIds] = useState(false)
  const [isLoadingImages, setIsLoadingImages] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const currentCellId = cellIds[currentIndex] ?? ''
  const cellCount = cellIds.length

  useEffect(() => {
    if (!requestedCellId || cellIds.length === 0) return
    const nextIndex = cellIds.indexOf(requestedCellId)
<<<<<<< HEAD
    if (nextIndex >= 0) {
      setCurrentIndex(nextIndex)
    }
  }, [cellIds, requestedCellId])
=======
    if (nextIndex >= 0 && nextIndex !== currentIndex) {
      setCurrentIndex(nextIndex)
    }
  }, [cellIds, currentIndex, requestedCellId])
>>>>>>> 0973fa0d44f7f8b4308309d57c72269dcd05828d

  useEffect(() => {
    if (!dbName || !currentCellId) return
    if (searchParams.get('cell_id') === currentCellId) return
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('cell_id', currentCellId)
<<<<<<< HEAD
    nextParams.delete('cell')
=======
>>>>>>> 0973fa0d44f7f8b4308309d57c72269dcd05828d
    setSearchParams(nextParams, { replace: true })
  }, [currentCellId, dbName, searchParams, setSearchParams])

  const fetchLabelOptions = useCallback(async () => {
    if (!dbName) {
      setLabelOptions(['All'])
      return
    }
    try {
      const res = await fetch(
        `${apiBase}/get-manual-labels?dbname=${encodeURIComponent(dbName)}`,
        { headers: { accept: 'application/json' } },
      )
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const data = (await res.json()) as string[]
      const cleaned = Array.isArray(data)
        ? data
            .map((label) => String(label).trim())
            .filter((label) => label.length > 0)
        : []
      const unique = Array.from(new Set(cleaned))
      setLabelOptions(['All', ...sortLabels(unique)])
    } catch (err) {
      setLabelOptions(['All'])
    }
  }, [apiBase, dbName])

  useEffect(() => {
    void fetchLabelOptions()
  }, [fetchLabelOptions])

  useEffect(() => {
    if (!labelOptions.includes(selectedLabel)) {
      setSelectedLabel('All')
    }
  }, [labelOptions, selectedLabel])

  const fetchCellIds = useCallback(async () => {
    if (!dbName) {
      setError('Database is required.')
      setCellIds([])
      setCurrentIndex(0)
      return
    }
    setIsLoadingIds(true)
    setError(null)
    try {
      const params = new URLSearchParams({ dbname: dbName })
      const endpoint =
        selectedLabel === 'All' ? 'get-cell-ids' : 'get-cell-ids-by-label'
      if (selectedLabel !== 'All') {
        params.set('label', selectedLabel)
      }
      const res = await fetch(`${apiBase}/${endpoint}?${params.toString()}`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const data = (await res.json()) as string[]
      const sorted = sortCellIds(Array.isArray(data) ? data : [])
      setCellIds(sorted)
      setCurrentIndex(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load cell IDs')
      setCellIds([])
    } finally {
      setIsLoadingIds(false)
    }
  }, [apiBase, dbName, selectedLabel])

  useEffect(() => {
    void fetchCellIds()
  }, [fetchCellIds])

  useEffect(() => {
    if (!dbName || !currentCellId) {
      setImages({ ph: null, fluo1: null, fluo2: null })
      return
    }

    let isActive = true
    const fetchChannel = async (
      channel: ChannelKey,
    ): Promise<{ url: string | null; missing: boolean }> => {
      const endpoint =
        modificationMode === 'optical-boost' && (channel === 'fluo1' || channel === 'fluo2')
          ? 'get-cell-image-optical-boost'
          : 'get-cell-image'
      const params = new URLSearchParams({
        dbname: dbName,
        cell_id: currentCellId,
        image_type: channel,
        draw_contour: String(overlayOptions.contour),
        draw_scale_bar: String(overlayOptions.scale),
      })
      const res = await fetch(
        `${apiBase}/${endpoint}?${params.toString()}`,
        { headers: { accept: 'image/png' } },
      )
      if (!res.ok) {
        return {
          url: null,
          missing: res.status === 404 && (channel === 'fluo1' || channel === 'fluo2'),
        }
      }
      const blob = await res.blob()
      return { url: URL.createObjectURL(blob), missing: false }
    }

    const loadImages = async () => {
      setIsLoadingImages(true)
      setError(null)
      setImages({ ph: null, fluo1: null, fluo2: null })
      setMissingChannels({ ph: false, fluo1: false, fluo2: false })
      try {
        const results = await Promise.all(
          channels.map((channel) => fetchChannel(channel.key)),
        )
        if (!isActive) {
          results.forEach((result) => {
            if (result.url) URL.revokeObjectURL(result.url)
          })
          return
        }
        setImages({
          ph: results[0].url,
          fluo1: results[1].url,
          fluo2: results[2].url,
        })
        setMissingChannels({
          ph: results[0].missing,
          fluo1: results[1].missing,
          fluo2: results[2].missing,
        })
      } catch (err) {
        if (isActive) {
          setError(err instanceof Error ? err.message : 'Failed to load images')
          setImages({ ph: null, fluo1: null, fluo2: null })
          setMissingChannels({ ph: false, fluo1: false, fluo2: false })
        }
      } finally {
        if (isActive) setIsLoadingImages(false)
      }
    }

    void loadImages()
    return () => {
      isActive = false
    }
  }, [apiBase, dbName, currentCellId, overlayOptions, modificationMode])

  useEffect(() => {
    if (!dbName || !currentCellId) {
      setManualLabel(null)
      setManualLabelError(null)
      return
    }
    let isActive = true
    const loadLabel = async () => {
      setIsLoadingLabel(true)
      setManualLabelError(null)
      try {
        const res = await fetch(
          `${apiBase}/get-cell-label?dbname=${encodeURIComponent(
            dbName,
          )}&cell_id=${encodeURIComponent(currentCellId)}`,
          { headers: { accept: 'application/json' } },
        )
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const label = (await res.json()) as string
        if (isActive) {
          setManualLabel(normalizeManualLabel(label))
        }
      } catch (err) {
        if (isActive) {
          setManualLabel(null)
        }
      } finally {
        if (isActive) setIsLoadingLabel(false)
      }
    }

    void loadLabel()
    return () => {
      isActive = false
    }
  }, [apiBase, currentCellId, dbName])

  const manualLabelOptions = useMemo(() => {
    const baseOptions = labelOptions.filter((option) => option !== 'All')
    const normalized = normalizeManualLabel(manualLabel)
    if (normalized && !baseOptions.includes(normalized)) {
      return [normalized, ...baseOptions]
    }
    return baseOptions
  }, [labelOptions, manualLabel])

  const updateManualLabel = useCallback(
    async (nextLabel: string) => {
      if (!dbName || !currentCellId) return
      const trimmed = nextLabel.trim()
      if (!trimmed) return
      setIsUpdatingLabel(true)
      setManualLabelError(null)
      const previousLabel = manualLabel
      setManualLabel(trimmed)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          label: trimmed,
        })
        const res = await fetch(`${apiBase}/update-cell-label?${params.toString()}`, {
          method: 'PATCH',
          headers: { accept: 'application/json' },
        })
        if (!res.ok) {
          throw new Error(`Label update failed (${res.status})`)
        }
        await fetchLabelOptions()
      } catch (err) {
        setManualLabel(previousLabel)
        setManualLabelError(err instanceof Error ? err.message : 'Failed to update label')
      } finally {
        setIsUpdatingLabel(false)
      }
    },
    [apiBase, currentCellId, dbName, fetchLabelOptions, manualLabel],
  )

  useEffect(() => {
    if (!dbName || !currentCellId || contourMode !== 'contour') {
      setContourData([])
      setContourError(null)
      setIsLoadingContour(false)
      return
    }
    let isActive = true
    const loadContour = async () => {
      setIsLoadingContour(true)
      setContourError(null)
      setContourData([])
      try {
        const res = await fetch(
          `${apiBase}/get-cell-contour?dbname=${encodeURIComponent(
            dbName,
          )}&cell_id=${encodeURIComponent(currentCellId)}`,
          { headers: { accept: 'application/json' } },
        )
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const data = (await res.json()) as { contour?: number[][] }
        if (isActive) {
          setContourData(Array.isArray(data.contour) ? data.contour : [])
        }
      } catch (err) {
        if (isActive) {
          setContourData([])
          setContourError(err instanceof Error ? err.message : 'Failed to load contour')
        }
      } finally {
        if (isActive) setIsLoadingContour(false)
      }
    }

    void loadContour()
    return () => {
      isActive = false
    }
  }, [apiBase, contourMode, currentCellId, dbName, contourRefreshKey, heatmapChannel])

  useEffect(() => {
    if (!dbName || !currentCellId || contourMode !== 'replot') {
      setReplotUrl(null)
      setReplotError(null)
      setIsLoadingReplot(false)
      return
    }
    let isActive = true
    const loadReplot = async () => {
      setIsLoadingReplot(true)
      setReplotError(null)
      setReplotUrl(null)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          image_type: replotChannel,
          degree: '4',
          dark_mode: 'true',
        })
        const res = await fetch(`${apiBase}/get-cell-replot?${params.toString()}`, {
          headers: { accept: 'image/png' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (isActive) {
          setReplotUrl(url)
        } else {
          URL.revokeObjectURL(url)
        }
      } catch (err) {
        if (isActive) {
          setReplotUrl(null)
          setReplotError(err instanceof Error ? err.message : 'Failed to load replot')
        }
      } finally {
        if (isActive) setIsLoadingReplot(false)
      }
    }

    void loadReplot()
    return () => {
      isActive = false
    }
  }, [apiBase, contourMode, currentCellId, dbName, contourRefreshKey, replotChannel])

  useEffect(() => {
    const shouldLoadOverlay =
      contourMode === 'overlay' || contourMode === 'overlay-fluo'
    if (!dbName || !currentCellId || !shouldLoadOverlay) {
      setOverlayUrl(null)
      setOverlayError(null)
      setIsLoadingOverlay(false)
      return
    }
    let isActive = true
    const controller = new AbortController()

    const loadOverlay = async () => {
      setIsLoadingOverlay(true)
      setOverlayError(null)
      setOverlayUrl(null)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          draw_scale_bar: String(overlayOptions.scale),
          overlay_mode: contourMode === 'overlay-fluo' ? 'fluo' : 'ph',
        })
        const res = await fetch(`${apiBase}/get-cell-overlay?${params.toString()}`, {
          signal: controller.signal,
          headers: { accept: 'image/png' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (isActive) {
          setOverlayUrl(url)
        } else {
          URL.revokeObjectURL(url)
        }
      } catch (err) {
        if (isActive) {
          setOverlayError(err instanceof Error ? err.message : 'Failed to load overlay')
          setOverlayUrl(null)
        }
      } finally {
        if (isActive) setIsLoadingOverlay(false)
      }
    }

    void loadOverlay()
    return () => {
      isActive = false
      controller.abort()
    }
  }, [
    apiBase,
    contourMode,
    currentCellId,
    dbName,
    contourRefreshKey,
    overlayOptions.scale,
  ])

  useEffect(() => {
    if (!dbName || !currentCellId || contourMode !== 'heatmap') {
      setHeatmapUrl(null)
      setHeatmapError(null)
      setIsLoadingHeatmap(false)
      return
    }
    let isActive = true
    const loadHeatmap = async () => {
      setIsLoadingHeatmap(true)
      setHeatmapError(null)
      setHeatmapUrl(null)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          image_type: heatmapChannel,
          degree: '4',
        })
        const res = await fetch(`${apiBase}/get-cell-heatmap?${params.toString()}`, {
          headers: { accept: 'image/png' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (isActive) {
          setHeatmapUrl(url)
        } else {
          URL.revokeObjectURL(url)
        }
      } catch (err) {
        if (isActive) {
          setHeatmapUrl(null)
          setHeatmapError(err instanceof Error ? err.message : 'Failed to load heatmap')
        }
      } finally {
        if (isActive) setIsLoadingHeatmap(false)
      }
    }

    void loadHeatmap()
    return () => {
      isActive = false
    }
  }, [apiBase, contourMode, currentCellId, dbName, contourRefreshKey])

  useEffect(() => {
    if (!dbName || !currentCellId || contourMode !== 'map256') {
      setMap256Url(null)
      setMap256JetUrl(null)
      setMap256Error(null)
      setMap256JetError(null)
      setIsLoadingMap256(false)
      return
    }
    let isActive = true
    const loadMap256 = async () => {
      setIsLoadingMap256(true)
      setMap256Error(null)
      setMap256JetError(null)
      setMap256Url(null)
      setMap256JetUrl(null)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          image_type: map256Channel,
          degree: '4',
        })
        const fetchMap = async (endpoint: string) => {
          const res = await fetch(`${apiBase}/${endpoint}?${params.toString()}`, {
            headers: { accept: 'image/png' },
          })
          if (!res.ok) {
            throw new Error(`Request failed (${res.status})`)
          }
          const blob = await res.blob()
          return URL.createObjectURL(blob)
        }
        const [rawResult, jetResult] = await Promise.allSettled([
          fetchMap('get-cell-map256'),
          fetchMap('get-cell-map256-jet'),
        ])
        if (!isActive) {
          if (rawResult.status === 'fulfilled') {
            URL.revokeObjectURL(rawResult.value)
          }
          if (jetResult.status === 'fulfilled') {
            URL.revokeObjectURL(jetResult.value)
          }
          return
        }
        if (rawResult.status === 'fulfilled') {
          setMap256Url(rawResult.value)
        } else {
          setMap256Error(
            rawResult.reason instanceof Error
              ? rawResult.reason.message
              : 'Failed to load map256',
          )
        }
        if (jetResult.status === 'fulfilled') {
          setMap256JetUrl(jetResult.value)
        } else {
          setMap256JetError(
            jetResult.reason instanceof Error
              ? jetResult.reason.message
              : 'Failed to load map256 jet',
          )
        }
      } catch (err) {
        if (isActive) {
          const message =
            err instanceof Error ? err.message : 'Failed to load map256'
          setMap256Url(null)
          setMap256JetUrl(null)
          setMap256Error(message)
          setMap256JetError(message)
        }
      } finally {
        if (isActive) setIsLoadingMap256(false)
      }
    }

    void loadMap256()
    return () => {
      isActive = false
    }
  }, [
    apiBase,
    contourMode,
    currentCellId,
    dbName,
    contourRefreshKey,
    map256Channel,
  ])

  useEffect(() => {
    if (!dbName || !currentCellId || contourMode !== 'distribution') {
      setDistributionUrl(null)
      setDistributionError(null)
      setIsLoadingDistribution(false)
      return
    }
    let isActive = true
    const loadDistribution = async () => {
      setIsLoadingDistribution(true)
      setDistributionError(null)
      setDistributionUrl(null)
      try {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          image_type: distributionChannel,
        })
        const res = await fetch(`${apiBase}/get-cell-distribution?${params.toString()}`, {
          headers: { accept: 'image/png' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        if (isActive) {
          setDistributionUrl(url)
        } else {
          URL.revokeObjectURL(url)
        }
      } catch (err) {
        if (isActive) {
          setDistributionUrl(null)
          setDistributionError(
            err instanceof Error ? err.message : 'Failed to load distribution',
          )
        }
      } finally {
        if (isActive) setIsLoadingDistribution(false)
      }
    }

    void loadDistribution()
    return () => {
      isActive = false
    }
  }, [
    apiBase,
    contourMode,
    currentCellId,
    dbName,
    contourRefreshKey,
    distributionChannel,
  ])

  const handleApplyModification = async () => {
    if (!dbName || !currentCellId || isApplyingModification) return
    setIsApplyingModification(true)
    setModificationError(null)
    try {
      if (modificationMode === 'elastic') {
        const params = new URLSearchParams({
          dbname: dbName,
          cell_id: currentCellId,
          delta: String(elasticDelta),
        })
        const res = await fetch(`${apiBase}/elastic-contour?${params.toString()}`, {
          method: 'PATCH',
          headers: { accept: 'application/json' },
        })
        if (!res.ok) {
          throw new Error(`Request failed (${res.status})`)
        }
        setContourRefreshKey((prev) => prev + 1)
      }
    } catch (err) {
      setModificationError(
        err instanceof Error ? err.message : 'Failed to apply modification',
      )
    } finally {
      setIsApplyingModification(false)
    }
  }

  useEffect(() => {
    return () => {
      if (replotUrl) {
        URL.revokeObjectURL(replotUrl)
      }
    }
  }, [replotUrl])

  useEffect(() => {
    return () => {
      if (overlayUrl) {
        URL.revokeObjectURL(overlayUrl)
      }
    }
  }, [overlayUrl])

  useEffect(() => {
    return () => {
      if (heatmapUrl) {
        URL.revokeObjectURL(heatmapUrl)
      }
    }
  }, [heatmapUrl])

  useEffect(() => {
    return () => {
      if (map256Url) {
        URL.revokeObjectURL(map256Url)
      }
    }
  }, [map256Url])

  useEffect(() => {
    return () => {
      if (map256JetUrl) {
        URL.revokeObjectURL(map256JetUrl)
      }
    }
  }, [map256JetUrl])

  useEffect(() => {
    return () => {
      if (distributionUrl) {
        URL.revokeObjectURL(distributionUrl)
      }
    }
  }, [distributionUrl])

  useEffect(() => {
    if (!images.ph) {
      setImageDimensions(null)
      return
    }
    let isActive = true
    const img = new Image()
    img.onload = () => {
      if (isActive) {
        const dims = {
          width: img.naturalWidth || img.width,
          height: img.naturalHeight || img.height,
        }
        setImageDimensions(dims)
      }
    }
    img.onerror = () => {
      if (isActive) {
        setImageDimensions(null)
      }
    }
    img.src = images.ph
    return () => {
      isActive = false
    }
  }, [images.ph])

  useEffect(() => {
    return () => {
      Object.values(images).forEach((url) => {
        if (url) URL.revokeObjectURL(url)
      })
    }
  }, [images])

  const handlePrevious = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0))
  }

  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, Math.max(cellCount - 1, 0)))
  }

  const isNavigatorDisabled = cellCount === 0 || isLoadingIds
  const isOverlayMode = contourMode === 'overlay' || contourMode === 'overlay-fluo'
  const contourPanelError =
    contourMode === 'replot'
      ? replotError
      : isOverlayMode
        ? overlayError
        : contourMode === 'heatmap'
          ? heatmapError
          : contourMode === 'map256'
            ? map256Error || map256JetError
          : contourMode === 'distribution'
            ? distributionError
          : contourError
  const isContourPending =
    contourMode === 'contour' &&
    (isLoadingContour || (contourData.length > 0 && !imageDimensions))
  const isOverlayPending = isOverlayMode && isLoadingOverlay
  const isHeatmapPending = contourMode === 'heatmap' && isLoadingHeatmap
  const isMap256Pending = contourMode === 'map256' && isLoadingMap256
  const isDistributionPending =
    contourMode === 'distribution' && isLoadingDistribution

  const contourView = useMemo(() => {
    const points = contourData
      .filter((pt) => Array.isArray(pt) && pt.length >= 2)
      .map(([x, y]) => [Number(x), Number(y)] as const)
      .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
    if (points.length === 0) return null
    const pathSegments = points.map(([x, y]) => `${x} ${y}`).join(' L ')
    const path = `M ${pathSegments}${points.length > 1 ? ' Z' : ''}`

    if (!imageDimensions) {
      return null
    }
    return {
      viewBox: `0 0 ${imageDimensions.width} ${imageDimensions.height}`,
      bounds: {
        minX: 0,
        minY: 0,
        width: imageDimensions.width,
        height: imageDimensions.height,
      },
      path,
    }
  }, [contourData, imageDimensions])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target) {
        const tagName = target.tagName
        if (
          tagName === 'INPUT' ||
          tagName === 'TEXTAREA' ||
          tagName === 'SELECT' ||
          target.isContentEditable
        ) {
          return
        }
      }
      if (isNavigatorDisabled) return

      if (event.key === 'Enter') {
        event.preventDefault()
        setCurrentIndex((prev) => Math.min(prev + 1, Math.max(cellCount - 1, 0)))
      } else if (event.code === 'Space' || event.key === ' ') {
        event.preventDefault()
        setCurrentIndex((prev) => Math.max(prev - 1, 0))
      } else if (event.key === 'n' || event.key === 'N') {
        event.preventDefault()
        void updateManualLabel('N/A')
      } else if (event.key === '1') {
        event.preventDefault()
        void updateManualLabel('1')
      } else if (event.key === '2') {
        event.preventDefault()
        void updateManualLabel('2')
      } else if (event.key === '3') {
        event.preventDefault()
        void updateManualLabel('3')
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [cellCount, isNavigatorDisabled, updateManualLabel])

  return (
    <Box
      minH="100vh"
      h={{ base: 'auto', lg: '100vh' }}
      bg="sand.50"
      color="ink.900"
      display="flex"
      flexDirection="column"
      overflow={{ base: 'visible', lg: 'hidden' }}
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
            Viewer
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
                <BreadcrumbCurrentLink color="ink.900">Cells</BreadcrumbCurrentLink>
              </BreadcrumbItem>
            </BreadcrumbList>
          </BreadcrumbRoot>
          <ReloadButton />
          <ThemeToggle />
        </HStack>
      </AppHeader>

      <Container
        maxW="80rem"
        py={{ base: 6, md: 4, lg: 3 }}
        flex="1"
        display="flex"
        flexDirection="column"
      >
        <Stack spacing={{ base: 5, lg: 4 }} flex="1" minH="0">
          <Box
            borderRadius="xl"
            bg="sand.100"
            border="1px solid"
            borderColor="sand.200"
            p="3"
          >
            <Stack spacing="3">
              <HStack spacing="2" justify="space-between" flexWrap="wrap">
                <Text fontSize="xs" letterSpacing="0.18em" color="ink.700">
                  Cell Control Panel
                </Text>
                <Text fontSize="xs" color="ink.700">
                  Database: {dbName || 'Not selected'}
                </Text>
              </HStack>
              <Grid
                templateColumns={{ base: 'minmax(0, 1fr)', md: 'repeat(3, minmax(0, 1fr))' }}
                gap="3"
              >
                <Box>
                  <Stack spacing="1">
                    <HStack spacing="2" align="flex-start" flexWrap="wrap">
                      <Box>
                        <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                          Label Filter
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={selectedLabel}
                            onChange={(event) => setSelectedLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            w="7rem"
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
                      </Box>
                      <Box>
                        <Text fontSize="xs" color="ink.700" mb="1">
                          Manual label
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={manualLabel ?? ''}
                            onChange={(event) => updateManualLabel(event.target.value)}
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h="2.25rem"
                            w="8rem"
                            color="ink.900"
                            isDisabled={
                              !dbName || !currentCellId || isLoadingLabel || isUpdatingLabel
                            }
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            <option value="" disabled>
                              Not set
                            </option>
                            {manualLabelOptions.map((option) => (
                              <option key={option} value={option}>
                                {option}
                              </option>
                            ))}
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                      </Box>
                    </HStack>
                    {manualLabelError && (
                      <Text fontSize="xs" color="violet.300">
                        {manualLabelError}
                      </Text>
                    )}
                  </Stack>
                </Box>
                <Box>
                  <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                    Overlays
                  </Text>
                  <HStack spacing="4" align="center" minH="2.25rem">
                    <Checkbox.Root
                      checked={overlayOptions.contour}
                      onCheckedChange={(details) =>
                        setOverlayOptions((prev) => ({
                          ...prev,
                          contour: details.checked,
                        }))
                      }
                      colorPalette="tide"
                      display="flex"
                      alignItems="center"
                      gap="2"
                    >
                      <Checkbox.HiddenInput />
                      <Checkbox.Control
                        borderColor="tide.400"
                        _checked={{ bg: 'tide.500', borderColor: 'tide.500', color: 'ink.900' }}
                      />
                      <Checkbox.Label fontSize="sm" color="ink.700">
                        Contour
                      </Checkbox.Label>
                    </Checkbox.Root>
                    <Checkbox.Root
                      checked={overlayOptions.scale}
                      onCheckedChange={(details) =>
                        setOverlayOptions((prev) => ({
                          ...prev,
                          scale: details.checked,
                        }))
                      }
                      colorPalette="tide"
                      display="flex"
                      alignItems="center"
                      gap="2"
                    >
                      <Checkbox.HiddenInput />
                      <Checkbox.Control
                        borderColor="tide.400"
                        _checked={{ bg: 'tide.500', borderColor: 'tide.500', color: 'ink.900' }}
                      />
                      <Checkbox.Label fontSize="sm" color="ink.700">
                        Scale
                      </Checkbox.Label>
                    </Checkbox.Root>
                  </HStack>
                </Box>
                <Box>
                  <Text fontSize="xs" letterSpacing="0.18em" color="ink.700" mb="1">
                    Navigator
                  </Text>
                  <Stack spacing="2">
                    <Grid templateColumns="repeat(3, minmax(0, 1fr))" gap="2">
                      <Box>
                        <Text fontSize="xs" color="ink.700">
                          Current cell
                        </Text>
                        <Text fontSize="sm" fontWeight="600">
                          {currentCellId || '-'}
                        </Text>
                      </Box>
                      <Box>
                        <Text fontSize="xs" color="ink.700">
                          Manual label
                        </Text>
                        <Text fontSize="sm" fontWeight="600">
                          {isLoadingLabel ? 'Loading...' : manualLabel ?? '-'}
                        </Text>
                      </Box>
                      <Box>
                        <Text fontSize="xs" color="ink.700">
                          Index
                        </Text>
                        <Text fontSize="sm" fontWeight="600">
                          {cellCount ? currentIndex + 1 : 0} / {cellCount}
                        </Text>
                      </Box>
                    </Grid>
                  </Stack>
                </Box>
              </Grid>
            </Stack>
          </Box>

          {error && (
            <Box px="4" py="3" bg="sand.100" borderRadius="md" border="1px solid" borderColor="sand.200">
              <Text fontSize="sm" color="violet.300">
                {error}
              </Text>
            </Box>
          )}

          {isLoadingIds && (
            <Text fontSize="sm" color="ink.700">
              Loading cell IDs...
            </Text>
          )}

          {!isLoadingIds && !error && cellCount === 0 && dbName && (
            <Text fontSize="sm" color="ink.700">
              No cells found in this database.
            </Text>
          )}

          <Grid
            templateColumns={{ base: 'minmax(0, 1fr)', lg: 'minmax(0, 1.6fr) minmax(0, 1fr)' }}
            gap="6"
            alignItems="stretch"
            flex="1"
            minH="0"
          >
            <Box
              overflowX={{ base: 'auto', lg: 'visible' }}
              pb={{ base: 2, lg: 0 }}
              h="100%"
              display="flex"
              flexDirection="column"
              gap={{ base: 3, lg: 2 }}
              minH="0"
            >
              <Box
                borderRadius="xl"
                bg="sand.100"
                border="1px dashed"
                borderColor="sand.200"
                p={{ base: 3, lg: 2 }}
                flex="0 0 auto"
                order={0}
              >
                <Stack spacing={{ base: 3, lg: 2 }}>
                  <Text fontSize="xs" letterSpacing="0.18em" color="ink.700">
                    Modification
                  </Text>
                  <HStack spacing={{ base: 3, lg: 2 }} align="flex-end" flexWrap="wrap">
                    <Box minW="10rem">
                      <Stack spacing="1">
                        <Text fontSize="xs" color="ink.700">
                          Mode
                        </Text>
                        <NativeSelect.Root>
                          <NativeSelect.Field
                            value={modificationMode}
                            onChange={(event) =>
                              setModificationMode(
                                event.target.value as 'elastic' | 'optical-boost',
                              )
                            }
                            bg="sand.50"
                            border="1px solid"
                            borderColor="sand.200"
                            fontSize="sm"
                            h={{ base: '2.25rem', lg: '2rem' }}
                            color="ink.900"
                            _focusVisible={{
                              borderColor: 'tide.400',
                              boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                            }}
                          >
                            <option value="elastic">Elastic contour</option>
                            <option value="optical-boost">Optical boost</option>
                          </NativeSelect.Field>
                          <NativeSelect.Indicator color="ink.700" />
                        </NativeSelect.Root>
                      </Stack>
                    </Box>
                    {modificationMode === 'elastic' && (
                      <>
                        <Box minW="9rem">
                          <Stack spacing="1">
                            <Text fontSize="xs" color="ink.700">
                              Elastic Δ
                            </Text>
                            <Input
                              type="number"
                              value={elasticDelta}
                              onChange={(event) => {
                                const next = Number(event.target.value)
                                setElasticDelta(Number.isNaN(next) ? 0 : next)
                              }}
                              min={-3}
                              max={3}
                              step={1}
                              bg="sand.50"
                              border="1px solid"
                              borderColor="sand.200"
                              fontSize="sm"
                              h={{ base: '2.25rem', lg: '2rem' }}
                              color="ink.900"
                              _focusVisible={{
                                borderColor: 'tide.400',
                                boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                              }}
                            />
                          </Stack>
                        </Box>
                        <Box minW="6rem" display="flex" alignItems="center">
                          <Button
                            size="sm"
                            h={{ base: '2.25rem', lg: '2rem' }}
                            bg="tide.500"
                            color="ink.900"
                            _hover={{ bg: 'tide.400' }}
                            onClick={handleApplyModification}
                            isDisabled={!dbName || !currentCellId || isApplyingModification}
                          >
                            {isApplyingModification ? 'Applying...' : 'Apply'}
                          </Button>
                        </Box>
                      </>
                    )}
                  </HStack>
                  {modificationError && (
                    <Text fontSize="xs" color="violet.300">
                      {modificationError}
                    </Text>
                  )}
                </Stack>
              </Box>
              <Grid
                templateColumns={{
                  base: 'repeat(3, 200px)',
                  md: 'repeat(3, minmax(0, 1fr))',
                }}
                gap={{ base: 3, lg: 2 }}
                minW={{ base: 'max-content', md: '0' }}
                order={1}
                flex="1"
                minH="0"
              >
                {channels.map((channel) => (
                  <Box
                    key={channel.key}
                    borderRadius="xl"
                    bg="sand.100"
                    border="1px solid"
                    borderColor="sand.200"
                    p={{ base: 3, lg: 2 }}
                  >
                    <Text
                      fontSize="xs"
                      letterSpacing="0.18em"
                      color="ink.700"
                      mb={{ base: 3, lg: 2 }}
                    >
                      {channel.label}
                    </Text>
                    <AspectRatio
                      ratio={1}
                      bg="sand.200"
                      borderRadius="lg"
                      overflow="hidden"
                    >
                      {isLoadingImages ? (
                        <Box display="flex" alignItems="center" justifyContent="center">
                          <Spinner color="ink.700" />
                        </Box>
                      ) : images[channel.key] ? (
                        <Box
                          as="img"
                          src={images[channel.key] ?? undefined}
                          alt={`${currentCellId} ${channel.label}`}
                          width="100%"
                          height="100%"
                          objectFit="contain"
                        />
                      ) : (
                        <Box display="flex" alignItems="center" justifyContent="center">
                          <Text fontSize="sm" color="ink.700">
                            {missingChannels[channel.key]
                              ? `${channel.key} does not exist.`
                              : 'Image not available.'}
                          </Text>
                        </Box>
                      )}
                    </AspectRatio>
                  </Box>
                ))}
              </Grid>
              <HStack
                spacing="3"
                align="center"
                order={2}
                flexWrap="wrap"
              >
                <Button
                  size="sm"
                  variant="outline"
                  borderColor="sand.200"
                  color="teal.500"
                  _hover={{ bg: 'sand.100', color: 'teal.600' }}
                  onClick={handlePrevious}
                  disabled={isNavigatorDisabled || currentIndex === 0}
                  gap="1"
                >
                  <ArrowLeft size={14} />
                  Previous
                </Button>
                <Box flex="1" minW="10rem">
                  <Slider.Root
                    value={[currentIndex]}
                    min={0}
                    max={Math.max(cellCount - 1, 0)}
                    step={1}
                    onValueChange={(details) => {
                      const next = details.value[0]
                      setCurrentIndex(typeof next === 'number' ? next : 0)
                    }}
                  >
                    <Slider.Control>
                      <Slider.Track bg="sand.200">
                        <Slider.Range bg="tide.400" />
                      </Slider.Track>
                      <Slider.Thumb index={0} />
                    </Slider.Control>
                  </Slider.Root>
                </Box>
                <Button
                  size="sm"
                  variant="outline"
                  borderColor="sand.200"
                  color="teal.500"
                  _hover={{ bg: 'sand.100', color: 'teal.600' }}
                  onClick={handleNext}
                  disabled={isNavigatorDisabled || currentIndex >= cellCount - 1}
                  gap="1"
                >
                  Next
                  <ArrowRight size={14} />
                </Button>
              </HStack>
            </Box>

            <Box
              borderRadius="xl"
              bg="sand.100"
              border="1px solid"
              borderColor="sand.200"
              p={{ base: 3, lg: 2 }}
              h="100%"
              display="flex"
              flexDirection="column"
            >
              <Stack spacing={{ base: 3, lg: 2 }} flex="1">
                <HStack justify="space-between" align="center" spacing="3" flexWrap="nowrap">
                  <Text fontSize="xs" letterSpacing="0.18em" color="ink.700">
                    Function Panel
                  </Text>
                  <HStack spacing="2" flexWrap="nowrap">
                    <Text fontSize="xs" color="ink.700" whiteSpace="nowrap">
                      Draw mode
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={contourMode}
                        onChange={(event) =>
                          setContourMode(
                            event.target.value as
                              | 'contour'
                              | 'replot'
                              | 'overlay'
                              | 'overlay-fluo'
                              | 'heatmap'
                              | 'map256'
                              | 'distribution',
                          )
                        }
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="xs"
                        h="2rem"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        <option value="contour">Contour</option>
                        <option value="replot">Replot</option>
                        <option value="overlay">Overlay</option>
                        <option value="overlay-fluo">Overlay Fluo</option>
                        <option value="heatmap">Heatmap</option>
                        <option value="map256">Map 256</option>
                        <option value="distribution">Distribution</option>
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </HStack>
                </HStack>
                {contourMode === 'heatmap' && (
                  <HStack spacing="2" align="center" flexWrap="nowrap">
                    <Text fontSize="xs" color="ink.700" whiteSpace="nowrap">
                      Channel
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={heatmapChannel}
                        onChange={(event) =>
                          setHeatmapChannel(event.target.value as 'fluo1' | 'fluo2')
                        }
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="xs"
                        h="2rem"
                        w="6rem"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        <option value="fluo1">Fluo1</option>
                        <option value="fluo2">Fluo2</option>
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </HStack>
                )}
                {contourMode === 'map256' && (
                  <HStack spacing="2" align="center" flexWrap="nowrap">
                    <Text fontSize="xs" color="ink.700" whiteSpace="nowrap">
                      Channel
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={map256Channel}
                        onChange={(event) =>
                          setMap256Channel(event.target.value as 'fluo1' | 'fluo2')
                        }
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="xs"
                        h="2rem"
                        w="6rem"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        <option value="fluo1">Fluo1</option>
                        <option value="fluo2">Fluo2</option>
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </HStack>
                )}
                {contourMode === 'distribution' && (
                  <HStack spacing="2" align="center" flexWrap="nowrap">
                    <Text fontSize="xs" color="ink.700" whiteSpace="nowrap">
                      Channel
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={distributionChannel}
                        onChange={(event) =>
                          setDistributionChannel(event.target.value as ChannelKey)
                        }
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="xs"
                        h="2rem"
                        w="6rem"
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
                  </HStack>
                )}
                {contourMode === 'replot' && (
                  <HStack spacing="2" align="center" flexWrap="nowrap">
                    <Text fontSize="xs" color="ink.700" whiteSpace="nowrap">
                      Channel
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={replotChannel}
                        onChange={(event) =>
                          setReplotChannel(event.target.value as ReplotChannel)
                        }
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        fontSize="xs"
                        h="2rem"
                        w="6rem"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        {replotChannelOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </HStack>
                )}
                <AspectRatio
                  ratio={1.1}
                  bg="sand.200"
                  borderRadius="lg"
                  overflow="hidden"
                >
                  <Box
                    display="flex"
                    alignItems="center"
                    justifyContent="center"
                    width="100%"
                    height="100%"
                  >
                    {contourMode === 'replot' ? (
                      isLoadingReplot ? (
                        <Spinner color="ink.700" />
                      ) : replotUrl ? (
                        <Box
                          as="img"
                          src={replotUrl ?? undefined}
                          alt={`${currentCellId} replot`}
                          width="100%"
                          height="100%"
                          objectFit="contain"
                        />
                      ) : (
                        <Text fontSize="sm" color="ink.700">
                          Replot not available.
                        </Text>
                      )
                    ) : isOverlayMode ? (
                      isOverlayPending ? (
                        <Spinner color="ink.700" />
                      ) : overlayUrl ? (
                        <Box
                          as="img"
                          src={overlayUrl ?? undefined}
                          alt={`${currentCellId} overlay`}
                          width="100%"
                          height="100%"
                          objectFit="contain"
                        />
                      ) : (
                        <Text fontSize="sm" color="ink.700">
                          Overlay not available.
                        </Text>
                      )
                    ) : contourMode === 'heatmap' ? (
                      isHeatmapPending ? (
                        <Spinner color="ink.700" />
                      ) : heatmapUrl ? (
                        <Box
                          as="img"
                          src={heatmapUrl ?? undefined}
                          alt={`${currentCellId} heatmap`}
                          width="100%"
                          height="100%"
                          objectFit="contain"
                        />
                      ) : (
                        <Text fontSize="sm" color="ink.700">
                          Heatmap not available.
                        </Text>
                      )
                    ) : contourMode === 'map256' ? (
                      <Box
                        width="100%"
                        height="100%"
                        p="2"
                        display="grid"
                        gridTemplateColumns="minmax(0, 1fr)"
                        gridTemplateRows="repeat(2, minmax(0, 1fr))"
                        gap="2"
                      >
                        <Box
                          display="flex"
                          flexDirection="column"
                          alignItems="center"
                          justifyContent="center"
                          minW="0"
                        >
                          <Text fontSize="xs" color="ink.700">
                            Map 256
                          </Text>
                          <Box
                            flex="1"
                            width="100%"
                            display="flex"
                            alignItems="center"
                            justifyContent="center"
                          >
                            {map256Url ? (
                              <Box
                                as="img"
                                src={map256Url ?? undefined}
                                alt={`${currentCellId} map256`}
                                width="100%"
                                height="100%"
                                objectFit="contain"
                              />
                            ) : isMap256Pending ? (
                              <Spinner color="ink.700" />
                            ) : (
                              <Text fontSize="xs" color="ink.700">
                                Map 256 not available.
                              </Text>
                            )}
                          </Box>
                        </Box>
                        <Box
                          display="flex"
                          flexDirection="column"
                          alignItems="center"
                          justifyContent="center"
                          minW="0"
                        >
                          <Text fontSize="xs" color="ink.700">
                            Jet
                          </Text>
                          <Box
                            flex="1"
                            width="100%"
                            display="flex"
                            alignItems="center"
                            justifyContent="center"
                          >
                            {map256JetUrl ? (
                              <Box
                                as="img"
                                src={map256JetUrl ?? undefined}
                                alt={`${currentCellId} map256 jet`}
                                width="100%"
                                height="100%"
                                objectFit="contain"
                              />
                            ) : isMap256Pending ? (
                              <Spinner color="ink.700" />
                            ) : (
                              <Text fontSize="xs" color="ink.700">
                                Jet not available.
                              </Text>
                            )}
                          </Box>
                        </Box>
                      </Box>
                    ) : contourMode === 'distribution' ? (
                      isDistributionPending ? (
                        <Spinner color="ink.700" />
                      ) : distributionUrl ? (
                        <Box
                          as="img"
                          src={distributionUrl ?? undefined}
                          alt={`${currentCellId} distribution`}
                          width="100%"
                          height="100%"
                          objectFit="contain"
                        />
                      ) : (
                        <Text fontSize="sm" color="ink.700">
                          Distribution not available.
                        </Text>
                      )
                    ) : isContourPending ? (
                      <Text fontSize="sm" color="ink.700">
                        Loading contour...
                      </Text>
                    ) : contourView ? (
                      <Box
                        as="svg"
                        key={`${currentCellId}-${contourMode}`}
                        width="100%"
                        height="100%"
                        viewBox={contourView.viewBox}
                        preserveAspectRatio="none"
                        role="img"
                        aria-label="Cell contour"
                      >
                        <path
                          d={contourView.path}
                          fill="none"
                          stroke="lime"
                          strokeWidth="1.5"
                          strokeLinejoin="round"
                          vectorEffect="non-scaling-stroke"
                        />
                      </Box>
                    ) : (
                      <Text fontSize="sm" color="ink.700">
                        No contour data.
                      </Text>
                    )}
                  </Box>
                </AspectRatio>
                {contourPanelError && (
                  <Text fontSize="sm" color="violet.300">
                    {contourPanelError}
                  </Text>
                )}
              </Stack>
            </Box>
          </Grid>
        </Stack>
      </Container>
    </Box>
  )
}
