import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Badge,
  Box,
  Button,
  Container,
  Flex,
  Grid,
  Heading,
  HStack,
  Icon,
  SimpleGrid,
  Stack,
  Text,
  VStack,
} from '@chakra-ui/react'
import { keyframes } from '@emotion/react'
import {
  Activity,
  ArrowRight,
  ChevronRight,
  Cpu,
  Database,
  Folder,
  Microscope,
  PencilLine,
  Server,
  Share2,
  Sparkles,
  Wifi,
  WifiOff,
  Workflow,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import PageHeader from '../components/PageHeader'
import ReloadButton, { runGitPullUpdate } from '../components/ReloadButton'
import ThemeToggleButton from '../components/ThemeToggleButton'
import { getApiBase } from '../utils/apiBase'

type StatusChipProps = {
  label: string
  value: string
  tone: 'ok' | 'error' | 'unknown'
  icon: LucideIcon
}

type MenuItem = {
  title: string
  description: string
  path: string
  icon: LucideIcon
  eyebrow: string
  accent: string
  external?: boolean
}

type ActivityPoint = {
  date: string
  count: number
}

type WorkflowStep = {
  title: string
  description: string
  icon: LucideIcon
}

const floatGlow = keyframes`
  0% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(-10px, 14px, 0) scale(1.04); }
  100% { transform: translate3d(0, 0, 0) scale(1); }
`

const pulseLine = keyframes`
  0%, 100% { opacity: 0.55; transform: scaleX(0.96); }
  50% { opacity: 1; transform: scaleX(1); }
`

const formatShortDate = (value: string) => {
  const parsed = new Date(`${value}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return `${parsed.getMonth() + 1}/${parsed.getDate()}`
}

const formatLongDate = (value: string) => {
  const parsed = new Date(`${value}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) {
    return value
  }
  return parsed.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  })
}

const StatusChip = ({ label, value, tone, icon }: StatusChipProps) => {
  const palette = {
    ok: {
      bg: 'rgba(15, 98, 90, 0.12)',
      color: 'tide.400',
      borderColor: 'rgba(34, 179, 161, 0.28)',
      dot: 'tide.400',
    },
    error: {
      bg: 'rgba(139, 92, 246, 0.10)',
      color: 'violet.400',
      borderColor: 'rgba(167, 139, 250, 0.28)',
      dot: 'violet.400',
    },
    unknown: {
      bg: 'sand.100',
      color: 'ink.700',
      borderColor: 'sand.200',
      dot: 'sand.300',
    },
  } as const
  const style = palette[tone]

  return (
    <HStack
      gap="2"
      px="3"
      py="2"
      borderRadius="full"
      bg={style.bg}
      border="1px solid"
      borderColor={style.borderColor}
      backdropFilter="blur(10px)"
    >
      <Box w="2px" h="14px" borderRadius="full" bg={style.dot} />
      <Icon as={icon} boxSize={3.5} color={style.color} />
      <Text fontSize="xs" fontWeight="600" color="ink.700">
        {label}
      </Text>
      <Text fontSize="xs" fontWeight="700" color={style.color} textTransform="uppercase" letterSpacing="0.08em">
        {value}
      </Text>
    </HStack>
  )
}

const ModuleCard = ({ item, onClick }: { item: MenuItem; onClick: () => void }) => (
  <Flex
    as="button"
    type="button"
    onClick={onClick}
    direction="column"
    align="stretch"
    gap="5"
    minH="220px"
    p="5"
    borderRadius="2xl"
    border="1px solid"
    borderColor="sand.200"
    bg="linear-gradient(180deg, rgba(255,255,255,0.9), rgba(245,251,255,0.78))"
    boxShadow="0 18px 48px rgba(15, 23, 42, 0.08)"
    transition="transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease"
    _hover={{
      transform: 'translateY(-4px)',
      borderColor: item.accent,
      boxShadow: '0 24px 60px rgba(15, 23, 42, 0.12)',
    }}
  >
    <Flex justify="space-between" align="start" gap="4">
      <VStack align="start" gap="3">
        <Badge
          alignSelf="start"
          bg="sand.100"
          color="ink.700"
          borderRadius="full"
          px="3"
          py="1"
          letterSpacing="0.14em"
          textTransform="uppercase"
          fontSize="0.65rem"
        >
          {item.eyebrow}
        </Badge>
        <Flex
          w="12"
          h="12"
          borderRadius="xl"
          align="center"
          justify="center"
          bg={item.accent}
          color="white"
          boxShadow="0 10px 20px rgba(15, 23, 42, 0.12)"
        >
          <Icon as={item.icon} boxSize={5} />
        </Flex>
      </VStack>
      <Flex
        w="10"
        h="10"
        borderRadius="full"
        align="center"
        justify="center"
        bg="sand.100"
        color="ink.700"
        border="1px solid"
        borderColor="sand.200"
      >
        <Icon as={ChevronRight} boxSize={4} />
      </Flex>
    </Flex>

    <Stack gap="2" flex="1" textAlign="left">
      <Heading size="md" color="ink.900" lineHeight="1.25">
        {item.title}
      </Heading>
      <Text fontSize="sm" color="ink.700" lineHeight="1.7">
        {item.description}
      </Text>
    </Stack>

    <Text
      fontSize="sm"
      fontWeight="600"
      color="ink.900"
      textAlign="left"
      display="inline-flex"
      alignItems="center"
      gap="2"
    >
      Open workspace
      <Icon as={ArrowRight} boxSize={4} />
    </Text>
  </Flex>
)

const workflowSteps: WorkflowStep[] = [
  {
    title: 'Ingest microscopy files',
    description: 'Review ND2 inputs and prepare extraction batches.',
    icon: Folder,
  },
  {
    title: 'Extract cells',
    description: 'Run segmentation and cell-level preprocessing from the top workspace.',
    icon: Cpu,
  },
  {
    title: 'Label phenotypes',
    description: 'Curate annotations and manage database-backed datasets.',
    icon: PencilLine,
  },
  {
    title: 'Analyze outcomes',
    description: 'Generate graphs, compare runs, and share interpretable outputs.',
    icon: Share2,
  },
]

const menuItems: MenuItem[] = [
  {
    title: 'Cell Extraction',
    description: 'Start from ND2 microscopy files and move directly into the extraction flow.',
    path: '/nd2files',
    icon: Microscope,
    eyebrow: 'Primary',
    accent: '#00AED3',
  },
  {
    title: 'Database Console',
    description: 'Inspect labeled cells, manage datasets, and keep the annotation backlog clean.',
    path: '/databases',
    icon: Database,
    eyebrow: 'Curation',
    accent: '#0f766e',
  },
  {
    title: 'File Manager',
    description: 'Browse local assets, verify uploads, and keep experiment folders organized.',
    path: '/files',
    icon: Folder,
    eyebrow: 'Storage',
    accent: '#2563eb',
  },
  {
    title: 'Graph Engine',
    description: 'Turn CSV outputs into plots and graph metrics for downstream review.',
    path: '/graph-engine',
    icon: Share2,
    eyebrow: 'Analysis',
    accent: '#7c3aed',
  },
  {
    title: 'Annotation Studio',
    description: 'Continue labeling work where phenotypes need manual review or correction.',
    path: '/annotation',
    icon: PencilLine,
    eyebrow: 'Review',
    accent: '#ea580c',
  },
  {
    title: 'ND2 Parser',
    description: 'Inspect raw ND2 payloads and validate metadata before processing.',
    path: '/nd2parser',
    icon: Workflow,
    eyebrow: 'Utilities',
    accent: '#0891b2',
  },
]

export default function TopPage() {
  const navigate = useNavigate()
  const apiBase = useMemo(() => getApiBase(), [])
  const [backendStatus, setBackendStatus] = useState<'ready' | 'error' | null>(null)
  const [internetStatus, setInternetStatus] = useState<boolean | null>(null)
  const [activityStatus, setActivityStatus] = useState<'idle' | 'loading' | 'error' | 'ready'>('idle')
  const [activityPoints, setActivityPoints] = useState<ActivityPoint[]>([])
  const topPageTrackedRef = useRef(false)

  const checkBackend = useCallback(async () => {
    if (!apiBase) {
      setBackendStatus(null)
      return
    }
    try {
      const res = await fetch(`${apiBase}/health`)
      setBackendStatus(res.ok ? 'ready' : 'error')
    } catch {
      setBackendStatus('error')
    }
  }, [apiBase])

  useEffect(() => {
    setInternetStatus(navigator.onLine)
    const updateOnline = () => setInternetStatus(true)
    const updateOffline = () => setInternetStatus(false)
    window.addEventListener('online', updateOnline)
    window.addEventListener('offline', updateOffline)
    return () => {
      window.removeEventListener('online', updateOnline)
      window.removeEventListener('offline', updateOffline)
    }
  }, [])

  useEffect(() => {
    checkBackend()
  }, [checkBackend])

  useEffect(() => {
    if (!apiBase) return
    void runGitPullUpdate(apiBase).catch((error) => {
      console.error('Auto update failed:', error)
    })
  }, [apiBase])

  useEffect(() => {
    if (!apiBase || topPageTrackedRef.current) return
    topPageTrackedRef.current = true
    fetch(`${apiBase}/activity/track`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_name: 'top_page' }),
    }).catch(() => {})
  }, [apiBase])

  useEffect(() => {
    if (!apiBase) return
    let isMounted = true
    const controller = new AbortController()

    const loadActivity = async () => {
      setActivityStatus('loading')
      try {
        const res = await fetch(`${apiBase}/activity/weekly?days=7`, {
          signal: controller.signal,
        })
        if (!res.ok) {
          throw new Error('Failed to load activity')
        }
        const data = await res.json()
        if (!isMounted) return
        const points = Array.isArray(data?.points) ? data.points : []
        setActivityPoints(points)
        setActivityStatus('ready')
      } catch {
        if (!isMounted || controller.signal.aborted) return
        setActivityPoints([])
        setActivityStatus('error')
      }
    }

    loadActivity()

    return () => {
      isMounted = false
      controller.abort()
    }
  }, [apiBase])

  const handleNavigate = (path: string, external?: boolean) => {
    if (external) {
      window.open(path, '_blank', 'noopener,noreferrer')
      return
    }
    navigate(path)
  }

  const backendTone: StatusChipProps['tone'] =
    backendStatus === 'ready' ? 'ok' : backendStatus === 'error' ? 'error' : 'unknown'
  const internetTone: StatusChipProps['tone'] =
    internetStatus === null ? 'unknown' : internetStatus ? 'ok' : 'error'

  const activityTotal = useMemo(
    () => activityPoints.reduce((sum, point) => sum + point.count, 0),
    [activityPoints],
  )
  const activityAverage = useMemo(() => {
    if (!activityPoints.length) return 0
    return Number((activityTotal / activityPoints.length).toFixed(1))
  }, [activityPoints.length, activityTotal])
  const activityPeak = useMemo(() => {
    if (!activityPoints.length) return null
    return activityPoints.reduce((max, point) => (point.count > max.count ? point : max))
  }, [activityPoints])

  const activityRangeLabel = useMemo(() => {
    if (!activityPoints.length) return 'Last 7 days'
    return `${formatLongDate(activityPoints[0].date)} - ${formatLongDate(
      activityPoints[activityPoints.length - 1].date,
    )}`
  }, [activityPoints])

  const activityLabels = useMemo(
    () => activityPoints.map((point) => formatShortDate(point.date)),
    [activityPoints],
  )

  const activityChart = useMemo(() => {
    if (!activityPoints.length) return null
    const width = 640
    const height = 200
    const paddingX = 24
    const paddingY = 24
    const innerWidth = width - paddingX * 2
    const innerHeight = height - paddingY * 2
    const maxCount = Math.max(...activityPoints.map((point) => point.count), 1)
    const slots = Math.max(activityPoints.length - 1, 1)
    const step = innerWidth / slots
    const offset = activityPoints.length === 1 ? innerWidth / 2 : 0

    const coordinates = activityPoints.map((point, index) => ({
      x: paddingX + offset + step * index,
      y: paddingY + innerHeight - (point.count / maxCount) * innerHeight,
      count: point.count,
      date: point.date,
    }))

    const lineSegment = coordinates.map((point) => `${point.x} ${point.y}`).join(' L ')
    const linePath = lineSegment ? `M ${lineSegment}` : ''
    const baselineY = paddingY + innerHeight
    const firstX = coordinates[0].x
    const lastX = coordinates[coordinates.length - 1].x
    const areaPath = lineSegment
      ? `M ${firstX} ${baselineY} L ${lineSegment} L ${lastX} ${baselineY} Z`
      : ''
    const gridLines = [0, 0.5, 1].map((ratio) => ({ y: paddingY + innerHeight - ratio * innerHeight }))

    return {
      width,
      height,
      paddingX,
      paddingY,
      coordinates,
      linePath,
      areaPath,
      gridLines,
    }
  }, [activityPoints])

  const heroStats = [
    {
      label: 'Modules',
      value: `${menuItems.length}`,
      helper: 'Core workspaces',
    },
    {
      label: '7-day actions',
      value: activityStatus === 'ready' ? `${activityTotal}` : '--',
      helper: activityRangeLabel,
    },
    {
      label: 'Peak day',
      value: activityPeak ? `${activityPeak.count}` : '--',
      helper: activityPeak ? formatLongDate(activityPeak.date) : 'Waiting for activity',
    },
  ]

  return (
    <Box
      minH="100vh"
      position="relative"
      overflow="hidden"
      bg="linear-gradient(180deg, #f8fcff 0%, #eef7fb 52%, #f8fcff 100%)"
      color="ink.900"
    >
      <Box
        position="absolute"
        top={{ base: '-80px', md: '-120px' }}
        right={{ base: '-100px', md: '-120px' }}
        w={{ base: '220px', md: '360px' }}
        h={{ base: '220px', md: '360px' }}
        borderRadius="full"
        bg="radial-gradient(circle, rgba(0,174,211,0.24) 0%, rgba(0,174,211,0.03) 58%, rgba(0,174,211,0) 72%)"
        filter="blur(8px)"
        animation={`${floatGlow} 14s ease-in-out infinite`}
        pointerEvents="none"
      />
      <Box
        position="absolute"
        bottom={{ base: '-90px', md: '-140px' }}
        left={{ base: '-100px', md: '-120px' }}
        w={{ base: '240px', md: '380px' }}
        h={{ base: '240px', md: '380px' }}
        borderRadius="full"
        bg="radial-gradient(circle, rgba(15,118,110,0.20) 0%, rgba(15,118,110,0.02) 62%, rgba(15,118,110,0) 74%)"
        filter="blur(10px)"
        animation={`${floatGlow} 18s ease-in-out infinite`}
        pointerEvents="none"
      />
      <Box
        position="absolute"
        inset="0"
        bgImage="linear-gradient(to right, rgba(148, 163, 184, 0.08) 1px, transparent 1px), linear-gradient(to bottom, rgba(148, 163, 184, 0.08) 1px, transparent 1px)"
        bgSize="42px 42px"
        maskImage="linear-gradient(180deg, rgba(0,0,0,0.75), rgba(0,0,0,0.25))"
        pointerEvents="none"
      />

      <Box position="relative">
        <PageHeader
          bg="rgba(248, 252, 255, 0.82)"
          actions={
            <HStack gap="2">
              <ReloadButton />
              <ThemeToggleButton />
            </HStack>
          }
        />

        <Container maxW="1280px" px={{ base: 4, md: 8 }} py={{ base: 6, md: 10 }}>
          <Stack gap={{ base: 6, md: 8 }}>
            <Grid templateColumns={{ base: '1fr', xl: 'minmax(0, 1.55fr) minmax(320px, 0.95fr)' }} gap="6">
              <Box
                position="relative"
                overflow="hidden"
                borderRadius="3xl"
                border="1px solid"
                borderColor="rgba(0, 174, 211, 0.16)"
                bg="linear-gradient(145deg, rgba(255,255,255,0.96), rgba(241,249,253,0.86))"
                boxShadow="0 24px 80px rgba(15, 23, 42, 0.12)"
                px={{ base: 5, md: 8 }}
                py={{ base: 6, md: 8 }}
              >
                <Box
                  position="absolute"
                  inset="0"
                  bg="linear-gradient(120deg, rgba(0,174,211,0.14), rgba(255,255,255,0) 42%, rgba(15,118,110,0.08) 100%)"
                  pointerEvents="none"
                />
                <Stack position="relative" gap="8">
                  <Stack gap="4" maxW="720px">
                    <HStack gap="3" flexWrap="wrap">
                      <Badge
                        bg="rgba(0, 174, 211, 0.12)"
                        color="tide.500"
                        borderRadius="full"
                        px="3"
                        py="1"
                        letterSpacing="0.18em"
                        textTransform="uppercase"
                        fontSize="0.7rem"
                      >
                        Home Console
                      </Badge>
                      <HStack gap="2" color="ink.700">
                        <Icon as={Sparkles} boxSize={4} color="tide.500" />
                        <Text fontSize="sm" fontWeight="600">
                          Extraction, annotation, and graph analysis in one place
                        </Text>
                      </HStack>
                    </HStack>

                    <Heading
                      fontSize={{ base: '2.2rem', md: '3.35rem' }}
                      lineHeight={{ base: '1.12', md: '1.02' }}
                      letterSpacing="-0.04em"
                      maxW="11ch"
                    >
                      Microscopy workflows that stay readable under load.
                    </Heading>

                    <Text maxW="62ch" fontSize={{ base: 'md', md: 'lg' }} color="ink.700" lineHeight="1.85">
                      PhenoPixel 6.0 keeps the daily path compact: bring in ND2 files, extract cells,
                      label phenotypes, and move straight into graph outputs without bouncing between
                      disconnected tools.
                    </Text>

                    <Flex gap="3" flexWrap="wrap">
                      <Button
                        size="lg"
                        px="6"
                        borderRadius="full"
                        bg="tide.500"
                        color="white"
                        _hover={{ bg: 'tide.400' }}
                        onClick={() => handleNavigate('/nd2files')}
                      >
                        Open Cell Extraction
                      </Button>
                      <Button
                        size="lg"
                        px="6"
                        borderRadius="full"
                        bg="white"
                        color="ink.900"
                        border="1px solid"
                        borderColor="sand.200"
                        _hover={{ bg: 'sand.100' }}
                        onClick={() => handleNavigate('/databases')}
                      >
                        Jump to Database Console
                      </Button>
                    </Flex>

                    <Flex gap="3" flexWrap="wrap">
                      <StatusChip
                        label="Backend"
                        value={backendStatus ?? 'Checking'}
                        tone={backendTone}
                        icon={Server}
                      />
                      <StatusChip
                        label="Network"
                        value={internetStatus === null ? 'Checking' : internetStatus ? 'Online' : 'Offline'}
                        tone={internetTone}
                        icon={internetStatus ? Wifi : WifiOff}
                      />
                    </Flex>
                  </Stack>

                  <SimpleGrid columns={{ base: 1, md: 3 }} gap="4">
                    {heroStats.map((stat) => (
                      <Box
                        key={stat.label}
                        p="4"
                        borderRadius="2xl"
                        bg="rgba(255,255,255,0.82)"
                        border="1px solid"
                        borderColor="rgba(203, 213, 225, 0.55)"
                        boxShadow="0 10px 24px rgba(15, 23, 42, 0.08)"
                      >
                        <Text
                          fontSize="xs"
                          color="ink.700"
                          fontWeight="700"
                          letterSpacing="0.18em"
                          textTransform="uppercase"
                          mb="2"
                        >
                          {stat.label}
                        </Text>
                        <Heading size="lg" color="ink.900" mb="1">
                          {stat.value}
                        </Heading>
                        <Text fontSize="sm" color="ink.700">
                          {stat.helper}
                        </Text>
                      </Box>
                    ))}
                  </SimpleGrid>
                </Stack>
              </Box>

              <Stack gap="6">
                <Box
                  borderRadius="3xl"
                  border="1px solid"
                  borderColor="sand.200"
                  bg="rgba(255,255,255,0.82)"
                  boxShadow="0 20px 48px rgba(15, 23, 42, 0.10)"
                  p="6"
                >
                  <HStack justify="space-between" align="start" mb="5">
                    <Stack gap="1">
                      <Text
                        fontSize="xs"
                        fontWeight="700"
                        color="ink.700"
                        letterSpacing="0.18em"
                        textTransform="uppercase"
                      >
                        System Pulse
                      </Text>
                      <Heading size="md" color="ink.900">
                        Current workspace status
                      </Heading>
                    </Stack>
                    <Flex
                      w="12"
                      h="12"
                      borderRadius="full"
                      align="center"
                      justify="center"
                      bg="rgba(0,174,211,0.12)"
                      color="tide.500"
                    >
                      <Icon as={Activity} boxSize={5} />
                    </Flex>
                  </HStack>

                  <Stack gap="4">
                    <Box
                      p="4"
                      borderRadius="2xl"
                      bg="sand.100"
                      border="1px solid"
                      borderColor="sand.200"
                    >
                      <Text fontSize="sm" fontWeight="600" color="ink.900" mb="2">
                        API endpoint
                      </Text>
                      <Text fontSize="sm" color="ink.700" wordBreak="break-all">
                        {apiBase}
                      </Text>
                    </Box>

                    <SimpleGrid columns={{ base: 1, md: 2, xl: 1 }} gap="4">
                      <Box p="4" borderRadius="2xl" bg="white" border="1px solid" borderColor="sand.200">
                        <Text fontSize="xs" fontWeight="700" color="ink.700" letterSpacing="0.16em" textTransform="uppercase" mb="2">
                          Connectivity
                        </Text>
                        <Text fontSize="lg" fontWeight="700" color="ink.900">
                          {internetStatus === null ? 'Checking' : internetStatus ? 'Stable' : 'Offline'}
                        </Text>
                        <Text fontSize="sm" color="ink.700" mt="1">
                          {internetStatus === false
                            ? 'Remote syncs and update checks are unavailable.'
                            : 'Network access is available for remote operations.'}
                        </Text>
                      </Box>
                      <Box p="4" borderRadius="2xl" bg="white" border="1px solid" borderColor="sand.200">
                        <Text fontSize="xs" fontWeight="700" color="ink.700" letterSpacing="0.16em" textTransform="uppercase" mb="2">
                          Backend
                        </Text>
                        <Text fontSize="lg" fontWeight="700" color="ink.900">
                          {backendStatus === 'ready' ? 'Ready' : backendStatus === 'error' ? 'Unavailable' : 'Checking'}
                        </Text>
                        <Text fontSize="sm" color="ink.700" mt="1">
                          {backendStatus === 'ready'
                            ? 'Health checks passed and activity metrics can load.'
                            : 'UI remains usable even when the API is down.'}
                        </Text>
                      </Box>
                    </SimpleGrid>
                  </Stack>
                </Box>

                <Box
                  position="relative"
                  overflow="hidden"
                  borderRadius="3xl"
                  border="1px solid"
                  borderColor="sand.200"
                  bg="linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,251,255,0.86))"
                  boxShadow="0 20px 48px rgba(15, 23, 42, 0.10)"
                  p="6"
                >
                  <Box
                    position="absolute"
                    top="18px"
                    right="18px"
                    w="88px"
                    h="2px"
                    bg="linear-gradient(90deg, rgba(0,174,211,0.3), rgba(15,118,110,0.85))"
                    animation={`${pulseLine} 4.4s ease-in-out infinite`}
                  />

                  <Stack gap="5">
                    <Stack gap="1">
                      <Text
                        fontSize="xs"
                        fontWeight="700"
                        color="ink.700"
                        letterSpacing="0.18em"
                        textTransform="uppercase"
                      >
                        Workflow Lane
                      </Text>
                      <Heading size="md" color="ink.900">
                        Recommended order of work
                      </Heading>
                    </Stack>

                    <VStack align="stretch" gap="4">
                      {workflowSteps.map((step, index) => (
                        <HStack key={step.title} align="start" gap="4">
                          <Flex
                            w="11"
                            h="11"
                            borderRadius="xl"
                            align="center"
                            justify="center"
                            bg={index === 0 ? 'rgba(0,174,211,0.12)' : 'sand.100'}
                            color={index === 0 ? 'tide.500' : 'ink.700'}
                            border="1px solid"
                            borderColor="sand.200"
                            flexShrink={0}
                          >
                            <Icon as={step.icon} boxSize={4.5} />
                          </Flex>
                          <Stack gap="1" pt="1">
                            <Text fontSize="sm" fontWeight="700" color="ink.900">
                              {step.title}
                            </Text>
                            <Text fontSize="sm" color="ink.700" lineHeight="1.7">
                              {step.description}
                            </Text>
                          </Stack>
                        </HStack>
                      ))}
                    </VStack>
                  </Stack>
                </Box>
              </Stack>
            </Grid>

            <Stack gap="4">
              <Flex justify="space-between" align={{ base: 'start', md: 'end' }} gap="4" flexWrap="wrap">
                <Stack gap="1">
                  <Text
                    fontSize="xs"
                    fontWeight="700"
                    color="ink.700"
                    letterSpacing="0.18em"
                    textTransform="uppercase"
                  >
                    Workspaces
                  </Text>
                  <Heading size="lg" color="ink.900">
                    Start from the task, not the menu tree
                  </Heading>
                </Stack>
                <Text maxW="520px" fontSize="sm" color="ink.700" lineHeight="1.8">
                  The top page now surfaces the highest-traffic workspaces first so common microscopy
                  operations are one click away.
                </Text>
              </Flex>

              <SimpleGrid columns={{ base: 1, md: 2, xl: 3 }} gap="5">
                {menuItems.map((item) => (
                  <ModuleCard
                    key={item.title}
                    item={item}
                    onClick={() => handleNavigate(item.path, item.external)}
                  />
                ))}
              </SimpleGrid>
            </Stack>

            <Grid templateColumns={{ base: '1fr', xl: 'minmax(0, 1.35fr) minmax(320px, 0.9fr)' }} gap="6">
              <Box
                display="flex"
                flexDirection="column"
                borderRadius="3xl"
                border="1px solid"
                borderColor="sand.200"
                bg="rgba(255,255,255,0.84)"
                boxShadow="0 20px 48px rgba(15, 23, 42, 0.10)"
                p={{ base: 5, md: 6 }}
                minH="360px"
              >
                <HStack justify="space-between" gap="4" mb="6" flexWrap="wrap">
                  <Stack gap="1">
                    <HStack gap="2">
                      <Icon as={Activity} boxSize={5} color="tide.500" />
                      <Heading size="md" color="ink.900">
                        Weekly Activity Trends
                      </Heading>
                    </HStack>
                    <Text fontSize="sm" color="ink.700">
                      Track usage over the last seven days without leaving the landing page.
                    </Text>
                  </Stack>
                  <Badge bg="sand.100" color="ink.900" fontSize="xs" px="3" py="1.5" borderRadius="full">
                    {activityRangeLabel}
                  </Badge>
                </HStack>

                <Box flex="1" w="full" position="relative" minH="240px">
                  {activityStatus === 'loading' || activityStatus === 'idle' ? (
                    <Flex justify="center" align="center" h="full">
                      <Stack
                        align="start"
                        gap="3"
                        maxW="360px"
                        p="5"
                        borderRadius="2xl"
                        bg="sand.100"
                        border="1px solid"
                        borderColor="sand.200"
                      >
                        <Badge bg="white" color="ink.700" borderRadius="full" px="3" py="1">
                          Metrics loading
                        </Badge>
                        <Heading size="sm" color="ink.900">
                          Pulling recent activity
                        </Heading>
                        <Text fontSize="sm" color="ink.700" lineHeight="1.7">
                          The dashboard is waiting for the weekly activity feed and will draw the
                          trend line as soon as the backend responds.
                        </Text>
                      </Stack>
                    </Flex>
                  ) : activityStatus === 'error' ? (
                    <Flex justify="center" align="center" h="full">
                      <Stack
                        align="start"
                        gap="3"
                        maxW="400px"
                        p="5"
                        borderRadius="2xl"
                        bg="linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,251,255,0.86))"
                        border="1px solid"
                        borderColor="sand.200"
                        boxShadow="0 14px 36px rgba(15, 23, 42, 0.08)"
                      >
                        <Badge bg="rgba(139, 92, 246, 0.10)" color="violet.400" borderRadius="full" px="3" py="1">
                          Metrics offline
                        </Badge>
                        <Heading size="sm" color="ink.900">
                          Activity data unavailable
                        </Heading>
                        <Text fontSize="sm" color="ink.700" lineHeight="1.7">
                          The landing page stays usable even when the API is unavailable. Core
                          workspaces remain accessible from the cards above.
                        </Text>
                        <Flex gap="3" flexWrap="wrap">
                          <StatusChip
                            label="Backend"
                            value={backendStatus ?? 'Checking'}
                            tone={backendTone}
                            icon={Server}
                          />
                          <StatusChip
                            label="Network"
                            value={internetStatus === null ? 'Checking' : internetStatus ? 'Online' : 'Offline'}
                            tone={internetTone}
                            icon={internetStatus ? Wifi : WifiOff}
                          />
                        </Flex>
                      </Stack>
                    </Flex>
                  ) : activityChart ? (
                    <Box w="full" h="240px" position="relative">
                      <svg
                        width="100%"
                        height="100%"
                        viewBox={`0 0 ${activityChart.width} ${activityChart.height}`}
                        preserveAspectRatio="none"
                        style={{ overflow: 'visible' }}
                      >
                        <defs>
                          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#00AED3" stopOpacity={0.28} />
                            <stop offset="100%" stopColor="#00AED3" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        {activityChart.gridLines.map((line, index) => (
                          <line
                            key={`grid-${index}`}
                            x1={activityChart.paddingX}
                            x2={activityChart.width - activityChart.paddingX}
                            y1={line.y}
                            y2={line.y}
                            stroke="rgba(148, 163, 184, 0.45)"
                            strokeWidth="1"
                          />
                        ))}
                        {activityChart.areaPath && <path d={activityChart.areaPath} fill="url(#chartGradient)" />}
                        {activityChart.linePath && (
                          <path
                            d={activityChart.linePath}
                            fill="none"
                            stroke="#00AED3"
                            strokeWidth="3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            vectorEffect="non-scaling-stroke"
                          />
                        )}
                      </svg>

                      {activityChart.coordinates.map((point, index) => {
                        const leftPercent = (point.x / activityChart.width) * 100
                        const topPercent = (point.y / activityChart.height) * 100

                        return (
                          <Box key={`point-group-${index}`}>
                            <Box
                              position="absolute"
                              left={`${leftPercent}%`}
                              top={`${topPercent}%`}
                              transform="translate(-50%, -50%)"
                              w="12px"
                              h="12px"
                              bg="white"
                              border="3px solid"
                              borderColor="tide.500"
                              borderRadius="full"
                              boxShadow="0 8px 16px rgba(0, 174, 211, 0.18)"
                              zIndex={2}
                            />
                            <Text
                              position="absolute"
                              left={`${leftPercent}%`}
                              bottom="-28px"
                              transform="translateX(-50%)"
                              fontSize="xs"
                              fontWeight="600"
                              color="ink.700"
                              whiteSpace="nowrap"
                            >
                              {activityLabels[index] ?? formatShortDate(point.date)}
                            </Text>
                          </Box>
                        )
                      })}
                    </Box>
                  ) : (
                    <Flex justify="center" align="center" h="full">
                      <Stack
                        align="start"
                        gap="3"
                        maxW="360px"
                        p="5"
                        borderRadius="2xl"
                        bg="sand.100"
                        border="1px solid"
                        borderColor="sand.200"
                      >
                        <Badge bg="white" color="ink.700" borderRadius="full" px="3" py="1">
                          Waiting for traffic
                        </Badge>
                        <Heading size="sm" color="ink.900">
                          No activity recorded yet
                        </Heading>
                        <Text fontSize="sm" color="ink.700" lineHeight="1.7">
                          Weekly metrics will appear here after the first tracked actions are sent
                          from the application.
                        </Text>
                      </Stack>
                    </Flex>
                  )}
                </Box>
              </Box>

              <Stack gap="6">
                <Box
                  borderRadius="3xl"
                  border="1px solid"
                  borderColor="sand.200"
                  bg="rgba(255,255,255,0.84)"
                  boxShadow="0 20px 48px rgba(15, 23, 42, 0.10)"
                  p="6"
                >
                  <Stack gap="5">
                    <Stack gap="1">
                      <Text
                        fontSize="xs"
                        fontWeight="700"
                        color="ink.700"
                        letterSpacing="0.18em"
                        textTransform="uppercase"
                      >
                        Snapshot
                      </Text>
                      <Heading size="md" color="ink.900">
                        Operational summary
                      </Heading>
                    </Stack>

                    <SimpleGrid columns={{ base: 1, sm: 3, xl: 1 }} gap="4">
                      <Box p="4" borderRadius="2xl" bg="sand.100" border="1px solid" borderColor="sand.200">
                        <Text fontSize="xs" fontWeight="700" color="ink.700" letterSpacing="0.16em" textTransform="uppercase" mb="2">
                          Total actions
                        </Text>
                        <Heading size="lg" color="ink.900">
                          {activityTotal}
                        </Heading>
                      </Box>
                      <Box p="4" borderRadius="2xl" bg="sand.100" border="1px solid" borderColor="sand.200">
                        <Text fontSize="xs" fontWeight="700" color="ink.700" letterSpacing="0.16em" textTransform="uppercase" mb="2">
                          Daily average
                        </Text>
                        <Heading size="lg" color="ink.900">
                          {activityAverage}
                        </Heading>
                      </Box>
                      <Box p="4" borderRadius="2xl" bg="sand.100" border="1px solid" borderColor="sand.200">
                        <Text fontSize="xs" fontWeight="700" color="ink.700" letterSpacing="0.16em" textTransform="uppercase" mb="2">
                          Peak day
                        </Text>
                        <Heading size="lg" color="ink.900">
                          {activityPeak ? activityPeak.count : '--'}
                        </Heading>
                        <Text fontSize="sm" color="ink.700" mt="1">
                          {activityPeak ? formatLongDate(activityPeak.date) : 'No activity captured yet'}
                        </Text>
                      </Box>
                    </SimpleGrid>
                  </Stack>
                </Box>

                <Box
                  borderRadius="3xl"
                  border="1px solid"
                  borderColor="sand.200"
                  bg="rgba(255,255,255,0.84)"
                  boxShadow="0 20px 48px rgba(15, 23, 42, 0.10)"
                  p="6"
                >
                  <Stack gap="4">
                    <HStack gap="3">
                      <Flex
                        w="12"
                        h="12"
                        borderRadius="xl"
                        align="center"
                        justify="center"
                        bg="rgba(15,118,110,0.12)"
                        color="tide.500"
                      >
                        <Icon as={Workflow} boxSize={5} />
                      </Flex>
                      <Stack gap="1">
                        <Heading size="md" color="ink.900">
                          Suggested next move
                        </Heading>
                        <Text fontSize="sm" color="ink.700">
                          Most teams start with file intake, then move into labeling or graph review.
                        </Text>
                      </Stack>
                    </HStack>

                    <Stack gap="3">
                      <Button
                        justifyContent="space-between"
                        size="lg"
                        borderRadius="xl"
                        bg="sand.100"
                        color="ink.900"
                        border="1px solid"
                        borderColor="sand.200"
                        _hover={{ bg: 'white' }}
                        onClick={() => handleNavigate('/files')}
                      >
                        Browse incoming files
                        <Icon as={ArrowRight} boxSize={4} />
                      </Button>
                      <Button
                        justifyContent="space-between"
                        size="lg"
                        borderRadius="xl"
                        bg="sand.100"
                        color="ink.900"
                        border="1px solid"
                        borderColor="sand.200"
                        _hover={{ bg: 'white' }}
                        onClick={() => handleNavigate('/graph-engine')}
                      >
                        Open graph engine
                        <Icon as={ArrowRight} boxSize={4} />
                      </Button>
                    </Stack>
                  </Stack>
                </Box>
              </Stack>
            </Grid>
          </Stack>
        </Container>
      </Box>
    </Box>
  )
}
