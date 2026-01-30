import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router-dom'
import {
  Badge,
  Box,
  BreadcrumbCurrentLink,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbRoot,
  BreadcrumbSeparator,
  Checkbox,
  Button,
  Container,
  Flex,
  Heading,
  HStack,
  Input,
  NativeSelect,
  Stack,
  Text,
  Separator,
  Slider,
} from '@chakra-ui/react'
import AppHeader from '../components/AppHeader'
import ReloadButton from '../components/ReloadButton'
import { getApiBase } from '../utils/apiBase'

type ExtractCellsResponse = {
  num_tiff: number
  ulid: string
  databases: { frame_start: number; frame_end: number; db_name: string }[]
  nd2_stem: string
}

type ExtractCellsStartResponse = {
  job_id: string
  status: 'running' | 'queued'
}

type ExtractCellsJobStatusResponse = {
  job_id: string
  status: 'running' | 'completed' | 'failed'
  result?: ExtractCellsResponse
  error?: string
}

const DEFAULT_PARAM1 = 130
const DEFAULT_IMAGE_SIZE = 200

const normalizeIntInput = (value: string) => {
  const digitsOnly = value.replace(/[^\d]/g, '')
  if (!digitsOnly) return ''
  const trimmed = digitsOnly.replace(/^0+/, '')
  return trimmed === '' ? '0' : trimmed
}

const coerceInt = (value: string, fallback: number, min: number) => {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed)) {
    return fallback
  }
  return Math.max(parsed, min)
}

const layerOptions = [
  { value: 'single', label: 'Single Layer' },
  { value: 'dual', label: 'Dual Layer' },
  { value: 'dual(reversed)', label: 'Dual Layer (Reversed)' },
  { value: 'triple', label: 'Triple Layer' },
  { value: 'quad', label: 'Quad Layer' },
]

export default function CellExtractionPage() {
  const [searchParams] = useSearchParams()
  const queryFilename = searchParams.get('filename') ?? ''
  const apiBase = useMemo(() => getApiBase(), [])
  const filename = queryFilename
  const [layerMode, setLayerMode] = useState(layerOptions[1].value)
  const [param1Input, setParam1Input] = useState(String(DEFAULT_PARAM1))
  const [imageSizeInput, setImageSizeInput] = useState(String(DEFAULT_IMAGE_SIZE))
  const [autoAnnotation, setAutoAnnotation] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>(
    'idle',
  )
  const [hasExtracted, setHasExtracted] = useState(false)
  const [extractedDbName, setExtractedDbName] = useState<string | null>(null)
  const [extractionStem, setExtractionStem] = useState<string | null>(null)
  const [imageCount, setImageCount] = useState<number | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isImageLoading, setIsImageLoading] = useState(false)
  const [imageError, setImageError] = useState<string | null>(null)

  const handleSubmit = useCallback(async () => {
    if (!filename.trim()) {
      setError('Filename is required')
      return
    }
    const param1Value = coerceInt(param1Input, DEFAULT_PARAM1, 0)
    const imageSizeValue = coerceInt(imageSizeInput, DEFAULT_IMAGE_SIZE, 1)
    setIsSubmitting(true)
    setError(null)
    setJobId(null)
    setJobStatus('running')
    setExtractionStem(null)
    setImageCount(null)
    setCurrentIndex(0)
    setImageUrl(null)
    setImageError(null)
    try {
      const payload = {
        filename: filename.trim(),
        layer_mode: layerMode,
        param1: param1Value,
        image_size: imageSizeValue,
        auto_annotation: autoAnnotation,
      }
      const res = await fetch(`${apiBase}/extract-cells`, {
        method: 'POST',
        headers: {
          accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const data = (await res.json()) as ExtractCellsStartResponse
      if (!data.job_id) {
        throw new Error('Job id missing from response')
      }
      setJobId(data.job_id)
      setJobStatus(data.status === 'queued' ? 'running' : data.status)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start extraction')
      setIsSubmitting(false)
      setJobStatus('failed')
    }
  }, [apiBase, autoAnnotation, filename, imageSizeInput, layerMode, param1Input])

  useEffect(() => {
    if (!jobId || jobStatus !== 'running') return
    let cancelled = false

    const pollStatus = async () => {
      try {
        const res = await fetch(`${apiBase}/extract-cells/${encodeURIComponent(jobId)}`)
        if (!res.ok) {
          throw new Error(`Status check failed (${res.status})`)
        }
        const data = (await res.json()) as ExtractCellsJobStatusResponse
        if (cancelled) return
        if (data.status === 'completed') {
          if (!data.result) {
            setError('Extraction result missing')
            setJobStatus('failed')
            setIsSubmitting(false)
            return
          }
          setExtractionStem(data.result.nd2_stem)
          setHasExtracted(true)
          const nextDbName = data.result.databases?.[0]?.db_name?.trim()
          setExtractedDbName(nextDbName ? nextDbName : null)
          setJobStatus('completed')
          setIsSubmitting(false)
          return
        }
        if (data.status === 'failed') {
          setError(data.error ?? 'Extraction failed')
          setJobStatus('failed')
          setIsSubmitting(false)
          return
        }
        setJobStatus('running')
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : 'Failed to check extraction status')
        setJobStatus('failed')
        setIsSubmitting(false)
      }
    }

    void pollStatus()
    const intervalId = window.setInterval(pollStatus, 2000)
    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [apiBase, jobId, jobStatus])

  const fetchImageCount = useCallback(
    async (stem: string) => {
      try {
        const res = await fetch(
          `${apiBase}/get-extracted-image-count?folder=${encodeURIComponent(stem)}`,
        )
        if (!res.ok) {
          throw new Error(`Image count failed (${res.status})`)
        }
        const data = (await res.json()) as { count: number }
        setImageCount(Number.isFinite(data.count) ? data.count : 0)
      } catch (err) {
        setImageCount(0)
        setImageError(err instanceof Error ? err.message : 'Failed to load image count')
      }
    },
    [apiBase],
  )

  const fetchImage = useCallback(
    async (stem: string, index: number) => {
      setIsImageLoading(true)
      setImageError(null)
      try {
        const res = await fetch(
          `${apiBase}/get-extracted-image?folder=${encodeURIComponent(stem)}&n=${index}`,
        )
        if (!res.ok) {
          throw new Error(`Image load failed (${res.status})`)
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        setImageUrl(url)
      } catch (err) {
        setImageUrl(null)
        setImageError(err instanceof Error ? err.message : 'Failed to load image')
      } finally {
        setIsImageLoading(false)
      }
    },
    [apiBase],
  )

  useEffect(() => {
    if (!extractionStem) return
    void fetchImageCount(extractionStem)
  }, [extractionStem, fetchImageCount])

  useEffect(() => {
    if (!extractionStem || imageCount === null) return
    if (imageCount <= 0) {
      setImageUrl(null)
      return
    }
    void fetchImage(extractionStem, currentIndex)
  }, [currentIndex, extractionStem, fetchImage, imageCount])

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl)
    }
  }, [imageUrl])

  return (
    <Box h="100vh" bg="sand.50" color="ink.900" display="flex" flexDirection="column">
      <AppHeader>
        <HStack
          as={RouterLink}
          to="/"
          spacing="3"
          color="inherit"
          _hover={{ textDecoration: 'none' }}
        >
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
            ND2
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
                <BreadcrumbLink as={RouterLink} to="/nd2files">
                  ND2 Files
                </BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator>/</BreadcrumbSeparator>
              <BreadcrumbItem>
                <BreadcrumbCurrentLink color="ink.900">Cell Extraction</BreadcrumbCurrentLink>
              </BreadcrumbItem>
            </BreadcrumbList>
          </BreadcrumbRoot>
          <ReloadButton />
        </HStack>
      </AppHeader>

      <Container
        maxW="72.5rem"
        py={{ base: 4, md: 6 }}
        flex="1"
        display="flex"
        flexDirection="column"
      >
        <Stack spacing="4" flex="1" minH="0">
          <Box
            display="grid"
            gridTemplateColumns={{ base: '1fr', lg: 'minmax(0, 1.15fr) minmax(0, 0.95fr)' }}
            gap="6"
            alignItems="stretch"
            flex="1"
            minH="0"
          >
            <Box
              bg="sand.100"
              border="1px solid"
              borderColor="sand.200"
              borderRadius="xl"
              p={{ base: 4, md: 5 }}
              h="full"
              display="flex"
              flexDirection="column"
              minH="0"
            >
              <Stack spacing="3" flex="1" minH="0">
                <Stack spacing="1">
                  <Text fontWeight="600">Extraction Settings</Text>
                  <Text fontSize="sm" color="ink.700">
                    Configure the inputs required for the extraction pipeline.
                  </Text>
                </Stack>
                <Separator borderColor="sand.200" />
                <Box
                  display="grid"
                  gridTemplateColumns={{ base: '1fr', md: 'repeat(3, minmax(0, 1fr))' }}
                  gap="3"
                >
                  <Stack spacing="2" gridColumn={{ base: 'auto', md: '1 / -1' }}>
                    <Text fontSize="sm" color="ink.700">
                      ND2 filename
                    </Text>
                    <Input
                      value={filename}
                      readOnly
                      placeholder="example.nd2"
                      border="1px solid"
                      borderColor="sand.200"
                      bg="sand.50"
                      color="ink.900"
                      _placeholder={{ color: 'ink.700' }}
                      _readOnly={{ opacity: 0.8 }}
                      _focusVisible={{
                        borderColor: 'tide.400',
                        boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                      }}
                    />
                  </Stack>

                  <Stack spacing="2">
                    <Text fontSize="sm" color="ink.700">
                      Mode
                    </Text>
                    <NativeSelect.Root>
                      <NativeSelect.Field
                        value={layerMode}
                        onChange={(event) => setLayerMode(event.target.value)}
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        color="ink.900"
                        _focusVisible={{
                          borderColor: 'tide.400',
                          boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                        }}
                      >
                        {layerOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </NativeSelect.Field>
                      <NativeSelect.Indicator color="ink.700" />
                    </NativeSelect.Root>
                  </Stack>

                  <Stack spacing="2">
                    <Text fontSize="sm" color="ink.700">
                      Param1
                    </Text>
                    <Input
                      type="number"
                      min={0}
                      placeholder="1-255"
                      value={param1Input}
                      onChange={(event) =>
                        setParam1Input(normalizeIntInput(event.target.value))
                      }
                      onBlur={() =>
                        setParam1Input((current) =>
                          current === '' ? String(DEFAULT_PARAM1) : current,
                        )
                      }
                      border="1px solid"
                      borderColor="sand.200"
                      bg="sand.50"
                      color="ink.900"
                      _placeholder={{ color: 'ink.700' }}
                      _focusVisible={{
                        borderColor: 'tide.400',
                        boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                      }}
                    />
                    <Text fontSize="xs" color="ink.700">
                      画面暗め: 60-75
                    </Text>
                    <Text fontSize="xs" color="ink.700">
                      画面明るめ: 115-140
                    </Text>
                  </Stack>

                  <Stack spacing="2">
                    <Text fontSize="sm" color="ink.700">
                      Image Size
                    </Text>
                    <Input
                      type="number"
                      min={1}
                      value={imageSizeInput}
                      onChange={(event) =>
                        setImageSizeInput(normalizeIntInput(event.target.value))
                      }
                      onBlur={() =>
                        setImageSizeInput((current) =>
                          current === '' ? String(DEFAULT_IMAGE_SIZE) : current,
                        )
                      }
                      border="1px solid"
                      borderColor="sand.200"
                      bg="sand.50"
                      color="ink.900"
                      _focusVisible={{
                        borderColor: 'tide.400',
                        boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                      }}
                    />
                  </Stack>

                  <Stack spacing="2">
                    <Text fontSize="sm" color="ink.700">
                      Auto Annotation
                    </Text>
                    <Checkbox.Root
                      checked={autoAnnotation}
                      onCheckedChange={(details) =>
                        setAutoAnnotation(details.checked === true)
                      }
                      colorPalette="tide"
                      display="flex"
                      alignItems="center"
                      gap="2"
                    >
                      <Checkbox.HiddenInput />
                      <Checkbox.Control
                        borderColor="tide.400"
                        _checked={{
                          bg: 'tide.500',
                          borderColor: 'tide.500',
                          color: 'ink.900',
                        }}
                      />
                      <Checkbox.Label fontSize="sm" color="ink.700">
                        Auto Annotation (Beta)
                      </Checkbox.Label>
                    </Checkbox.Root>
                  </Stack>
                </Box>

                <HStack justify="flex-end" flexWrap="wrap" gap="2">
                  {hasExtracted && extractedDbName && (
                    <HStack spacing="2" mr="auto">
                      <Button
                        size="sm"
                        bg="sand.300"
                        color="ink.900"
                        _hover={{ bg: 'sand.200' }}
                        as={RouterLink}
                        to={`/databases?search_dbname=${encodeURIComponent(extractedDbName)}`}
                      >
                        Go to database
                      </Button>
                      <Button
                        size="sm"
                        bg="sand.300"
                        color="ink.900"
                        _hover={{ bg: 'sand.200' }}
                        as={RouterLink}
                        to={`/annotation?dbname=${encodeURIComponent(extractedDbName)}`}
                      >
                        Go to annotation
                      </Button>
                    </HStack>
                  )}
                  <Button
                    size="sm"
                    bg="tide.500"
                    color="ink.900"
                    _hover={{ bg: 'tide.400' }}
                    onClick={handleSubmit}
                    loading={isSubmitting}
                  >
                    {isSubmitting ? 'Extracting...' : hasExtracted ? 'Re-extract' : 'Extract Cells'}
                  </Button>
                </HStack>
              </Stack>
            </Box>

            <Box
              bg="sand.100"
              border="1px solid"
              borderColor="sand.200"
              borderRadius="xl"
              p={{ base: 4, md: 5 }}
              h="full"
              display="flex"
              flexDirection="column"
              minH="0"
            >
              <Stack spacing="3" flex="1" minH="0">
                <Stack spacing="1">
                  <Text fontWeight="600">Extraction Preview</Text>
                  {(imageCount === null || imageCount <= 0) && (
                    <Text fontSize="sm" color="ink.700">
                      {imageCount === null
                        ? 'Run extraction to preview 0.png.'
                        : 'No extracted images found.'}
                    </Text>
                  )}
                </Stack>

                <Box
                  bg="sand.200"
                  borderRadius="lg"
                  display="flex"
                  alignItems="center"
                  justifyContent="center"
                  overflow="hidden"
                  flex="1"
                  minH="0"
                >
                  {imageUrl ? (
                    <Box
                      as="img"
                      src={imageUrl}
                      alt="Extracted cell preview"
                      w="100%"
                      h="auto"
                      maxH="100%"
                      objectFit="contain"
                    />
                  ) : (
                    <Flex align="center" justify="center" color="ink.700" fontSize="sm">
                      {isImageLoading
                        ? 'Loading image...'
                        : imageError
                          ? imageError
                          : 'No preview available.'}
                    </Flex>
                  )}
                </Box>

                {imageCount !== null && imageCount > 0 && (
                  <Stack spacing="2">
                    <HStack justify="space-between">
                      <Text fontSize="xs" color="ink.700">
                        Frame {currentIndex}
                      </Text>
                      <Text fontSize="xs" color="ink.700">
                        / {Math.max(imageCount - 1, 0)}
                      </Text>
                    </HStack>
                    <Slider.Root
                      value={[currentIndex]}
                      min={0}
                      max={Math.max(imageCount - 1, 0)}
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
                  </Stack>
                )}
              </Stack>
            </Box>

          </Box>

          {error && (
            <Box
              bg="sand.100"
              border="1px solid"
              borderColor="violet.400"
              borderRadius="lg"
              px="4"
              py="3"
            >
              <Text fontSize="sm" color="violet.300">
                {error}
              </Text>
            </Box>
          )}
        </Stack>
      </Container>
    </Box>
  )
}
