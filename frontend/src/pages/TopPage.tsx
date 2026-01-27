import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Badge,
  Box,
  BreadcrumbCurrentLink,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbRoot,
  Container,
  Flex,
  Heading,
  HStack,
  Icon,
  SimpleGrid,
  Stack,
  Text,
  VStack,
} from '@chakra-ui/react'
import {
  Cpu,
  Database,
  Folder,
  Server,
  Wifi,
  WifiOff,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import AppHeader from '../components/AppHeader'
import ReloadButton from '../components/ReloadButton'
import ThemeToggle from '../components/ThemeToggle'
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
  accent: string
  external?: boolean
}

type ActivityPoint = {
  date: string
  count: number
}

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
    ok: { accent: 'tide.300', text: 'ink.900' },
    error: { accent: 'violet.500', text: 'ink.900' },
    unknown: { accent: 'sand.300', text: 'ink.700' },
  } as const
  const style = palette[tone]

  return (
    <HStack
      spacing="3"
      px="3"
      py="2"
      borderRadius="md"
      bg="sand.100"
      border="1px solid"
      borderColor="sand.200"
    >
      <Box w="0.5rem" h="0.5rem" borderRadius="full" bg={style.accent} />
      <Icon as={icon} boxSize={4} color={style.accent} />
      <Text fontSize="sm" color="ink.700">
        {label}
      </Text>
      <Text fontSize="sm" fontWeight="600" color={style.text}>
        {value}
      </Text>
    </HStack>
  )
}

const MenuCard = ({ item, onClick }: { item: MenuItem; onClick: () => void }) => (
  <Box
    role="button"
    onClick={onClick}
    bg="sand.100"
    border="1px solid"
    borderColor="sand.200"
    borderRadius="xl"
    p="4"
    minH="11.25rem"
    transition="transform 0.2s ease, border-color 0.2s ease"
    _hover={{
      transform: 'translateY(-0.125rem)',
      borderColor: item.accent,
    }}
  >
    <VStack align="start" spacing="3">
      <Flex
        w="2.75rem"
        h="2.75rem"
        borderRadius="lg"
        bg="sand.200"
        align="center"
        justify="center"
        border="1px solid"
        borderColor="sand.300"
        color={item.accent}
      >
        <Icon as={item.icon} boxSize={5} />
      </Flex>
      <Box>
        <Text fontWeight="600" mb="1">
          {item.title}
        </Text>
        <Text fontSize="sm" color="ink.700">
          {item.description}
        </Text>
      </Box>
    </VStack>
  </Box>
)

const menuItems: MenuItem[] = [
  {
    title: 'Cell Extraction',
    description: 'Extract cells from ND2 microscopy files.',
    path: '/nd2files',
    icon: Cpu,
    accent: 'tide.300',
  },
  {
    title: 'Database Console',
    description: 'Label cells and manage datasets.',
    path: '/databases',
    icon: Database,
    accent: 'violet.400',
  },
  {
    title: 'File Manager',
    description: 'Manage files on the local server.',
    path: '/files',
    icon: Folder,
    accent: 'violet.400',
  },
]

export default function TopPage() {
  const navigate = useNavigate()
  const apiBase = useMemo(() => getApiBase(), [])
  const [backendStatus, setBackendStatus] = useState<'ready' | 'error' | null>(
    null,
  )
  const [internetStatus, setInternetStatus] = useState<boolean | null>(null)
  const [activityStatus, setActivityStatus] = useState<
    'idle' | 'loading' | 'error' | 'ready'
  >('idle')
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
    } catch (error) {
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
    if (!apiBase || topPageTrackedRef.current) {
      return
    }
    topPageTrackedRef.current = true
    fetch(`${apiBase}/activity/track`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action_name: 'top_page' }),
    }).catch(() => {})
  }, [apiBase])

  useEffect(() => {
    if (!apiBase) {
      return
    }
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
        if (!isMounted) {
          return
        }
        const points = Array.isArray(data?.points) ? data.points : []
        setActivityPoints(points)
        setActivityStatus('ready')
      } catch (error) {
        if (!isMounted || controller.signal.aborted) {
          return
        }
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
    if (!activityPoints.length) {
      return 0
    }
    const average = activityTotal / activityPoints.length
    return Number(average.toFixed(1))
  }, [activityPoints.length, activityTotal])
  const activityPeak = useMemo(() => {
    if (!activityPoints.length) {
      return null
    }
    return activityPoints.reduce((max, point) =>
      point.count > max.count ? point : max,
    )
  }, [activityPoints])
  const activityRangeLabel = useMemo(() => {
    if (!activityPoints.length) {
      return 'Last 7 days'
    }
    const start = formatLongDate(activityPoints[0].date)
    const end = formatLongDate(activityPoints[activityPoints.length - 1].date)
    return `${start} - ${end}`
  }, [activityPoints])
  const activityLabels = useMemo(
    () => activityPoints.map((point) => formatShortDate(point.date)),
    [activityPoints],
  )
  const activityChart = useMemo(() => {
    if (!activityPoints.length) {
      return null
    }
    const width = 640
    const height = 180
    const paddingX = 28
    const paddingY = 18
    const innerWidth = width - paddingX * 2
    const innerHeight = height - paddingY * 2
    const maxCount = Math.max(
      ...activityPoints.map((point) => point.count),
      1,
    )
    const slots = Math.max(activityPoints.length - 1, 1)
    const step = innerWidth / slots
    const offset = activityPoints.length === 1 ? innerWidth / 2 : 0
    const coordinates = activityPoints.map((point, index) => {
      const x = paddingX + offset + step * index
      const y = paddingY + innerHeight - (point.count / maxCount) * innerHeight
      return { x, y, count: point.count, date: point.date }
    })
    const lineSegment = coordinates
      .map((point) => `${point.x} ${point.y}`)
      .join(' L ')
    const linePath = lineSegment ? `M ${lineSegment}` : ''
    const baselineY = paddingY + innerHeight
    const firstX = coordinates[0].x
    const lastX = coordinates[coordinates.length - 1].x
    const areaPath = lineSegment
      ? `M ${firstX} ${baselineY} L ${lineSegment} L ${lastX} ${baselineY} Z`
      : ''
    const gridLines = [0, 0.5, 1].map((ratio) => ({
      y: paddingY + innerHeight - ratio * innerHeight,
    }))
    return {
      width,
      height,
      paddingX,
      paddingY,
      innerWidth,
      innerHeight,
      maxCount,
      coordinates,
      linePath,
      areaPath,
      gridLines,
    }
  }, [activityPoints])

  return (
    <Box minH="100vh" bg="sand.50" color="ink.900" position="relative">
      <Box position="relative">
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
              beta
            </Badge>
          </HStack>
          <HStack spacing="4" align="center">
            <BreadcrumbRoot fontSize="sm" color="ink.700">
              <BreadcrumbList>
                <BreadcrumbItem>
                  <BreadcrumbCurrentLink color="ink.900">Dashboard</BreadcrumbCurrentLink>
                </BreadcrumbItem>
              </BreadcrumbList>
            </BreadcrumbRoot>
            <ReloadButton />
            <ThemeToggle />
          </HStack>
        </AppHeader>

        <Container maxW="72.5rem" py={{ base: 8, md: 12 }}>
          <Stack spacing={{ base: 8, md: 10 }}>
            <Box
              bg="sand.100"
              border="1px solid"
              borderColor="sand.200"
              borderRadius="xl"
              p={{ base: 4, md: 5 }}
            >
              <HStack justify="space-between" flexWrap="wrap" gap="4">
                <Text fontWeight="600">System Status</Text>
                <HStack spacing="3" flexWrap="wrap">
                  <StatusChip
                    label="Backend"
                    value={backendStatus ?? 'unknown'}
                    tone={backendTone}
                    icon={Server}
                  />
                  <StatusChip
                    label="Internet"
                    value={
                      internetStatus === null
                        ? 'unknown'
                        : internetStatus
                          ? 'connected'
                          : 'offline'
                    }
                    tone={internetTone}
                    icon={internetStatus ? Wifi : WifiOff}
                  />
                </HStack>
              </HStack>
            </Box>

            <Box
              bg="sand.100"
              border="1px solid"
              borderColor="sand.200"
              borderRadius="xl"
              p={{ base: 4, md: 5 }}
            >
              <Stack spacing="4">
                <HStack justify="space-between" flexWrap="wrap" gap="4">
                  <Box>
                    <Text
                      fontSize="xs"
                      color="ink.700"
                      textTransform="uppercase"
                      letterSpacing="0.2em"
                      mb="1"
                    >
                      Activity
                    </Text>
                    <Heading size="md">Weekly usage</Heading>
                    <Text fontSize="sm" color="ink.700">
                      Tracks top page visits, cell extraction, and bulk engine usage.
                    </Text>
                  </Box>
                  <Badge
                    bg="sand.200"
                    color="ink.900"
                    borderRadius="full"
                    px="3"
                    py="1"
                    fontSize="0.65rem"
                    letterSpacing="0.18em"
                    textTransform="uppercase"
                  >
                    {activityRangeLabel}
                  </Badge>
                </HStack>

                {(activityStatus === 'loading' || activityStatus === 'idle') && (
                  <Text fontSize="sm" color="ink.700">
                    Loading activity…
                  </Text>
                )}

                {activityStatus === 'error' && (
                  <Text fontSize="sm" color="violet.500">
                    Activity data is unavailable. Check the backend connection.
                  </Text>
                )}

                {activityStatus === 'ready' && (
                  <Stack spacing="4">
                    <SimpleGrid columns={{ base: 1, md: 3 }} gap="3">
                      <Box
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        borderRadius="lg"
                        p="3"
                      >
                        <Text
                          fontSize="xs"
                          color="ink.700"
                          textTransform="uppercase"
                          letterSpacing="0.16em"
                          mb="1"
                        >
                          Total actions
                        </Text>
                        <Text fontSize="2xl" fontWeight="600">
                          {activityTotal}
                        </Text>
                      </Box>
                      <Box
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        borderRadius="lg"
                        p="3"
                      >
                        <Text
                          fontSize="xs"
                          color="ink.700"
                          textTransform="uppercase"
                          letterSpacing="0.16em"
                          mb="1"
                        >
                          Peak day
                        </Text>
                        <Text fontSize="xl" fontWeight="600">
                          {activityPeak ? activityPeak.count : 0}
                        </Text>
                        <Text fontSize="xs" color="ink.700">
                          {activityPeak ? formatLongDate(activityPeak.date) : 'No data'}
                        </Text>
                      </Box>
                      <Box
                        bg="sand.50"
                        border="1px solid"
                        borderColor="sand.200"
                        borderRadius="lg"
                        p="3"
                      >
                        <Text
                          fontSize="xs"
                          color="ink.700"
                          textTransform="uppercase"
                          letterSpacing="0.16em"
                          mb="1"
                        >
                          Daily avg
                        </Text>
                        <Text fontSize="2xl" fontWeight="600">
                          {activityAverage}
                        </Text>
                      </Box>
                    </SimpleGrid>

                    {activityChart ? (
                      <>
                        <Box
                          bg="sand.50"
                          border="1px solid"
                          borderColor="sand.200"
                          borderRadius="lg"
                          p="3"
                        >
                          <Box w="full" h={{ base: '180px', md: '200px' }}>
                            <svg
                              width="100%"
                              height="100%"
                              viewBox={`0 0 ${activityChart.width} ${activityChart.height}`}
                              preserveAspectRatio="xMidYMid meet"
                              role="img"
                              aria-label="Weekly activity line chart"
                            >
                              <defs>
                                <linearGradient
                                  id="activityGradient"
                                  x1="0"
                                  y1="0"
                                  x2="0"
                                  y2="1"
                                >
                                  <stop
                                    offset="0%"
                                    stopColor="rgba(45, 212, 191, 0.35)"
                                  />
                                  <stop
                                    offset="100%"
                                    stopColor="rgba(45, 212, 191, 0)"
                                  />
                                </linearGradient>
                              </defs>
                              {activityChart.gridLines.map((line, index) => (
                                <line
                                  key={`grid-${index}`}
                                  x1={activityChart.paddingX}
                                  x2={activityChart.width - activityChart.paddingX}
                                  y1={line.y}
                                  y2={line.y}
                                  stroke="var(--chakra-colors-sand-300)"
                                  strokeDasharray="4 6"
                                />
                              ))}
                              {activityChart.areaPath && (
                                <path
                                  d={activityChart.areaPath}
                                  fill="url(#activityGradient)"
                                />
                              )}
                              {activityChart.linePath && (
                                <path
                                  d={activityChart.linePath}
                                  fill="none"
                                  stroke="var(--chakra-colors-tide-400)"
                                  strokeWidth="2.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              )}
                              {activityChart.coordinates.map((point, index) => (
                                <circle
                                  key={`point-${index}`}
                                  cx={point.x}
                                  cy={point.y}
                                  r="3.5"
                                  fill="var(--chakra-colors-tide-400)"
                                  stroke="var(--chakra-colors-sand-50)"
                                  strokeWidth="1.5"
                                />
                              ))}
                              {activityChart.coordinates.map((point, index) => (
                                <text
                                  key={`label-${index}`}
                                  x={point.x}
                                  y={activityChart.height - activityChart.paddingY / 2}
                                  textAnchor="middle"
                                  dominantBaseline="middle"
                                  fontSize="10"
                                  fill="var(--chakra-colors-ink-700)"
                                >
                                  {activityLabels[index] ?? formatShortDate(point.date)}
                                </text>
                              ))}
                            </svg>
                          </Box>
                        </Box>
                      </>
                    ) : (
                      <Text fontSize="sm" color="ink.700">
                        No activity data yet.
                      </Text>
                    )}

                    {activityTotal === 0 && (
                      <Text fontSize="sm" color="ink.700">
                        Start by extracting cells or running a bulk analysis to
                        see usage trends here.
                      </Text>
                    )}
                  </Stack>
                )}
              </Stack>
            </Box>

            <Stack spacing="4">
              <SimpleGrid
                columns={{ base: 1, sm: 2, lg: 3 }}
                columnGap={{ base: 4, md: 6, lg: 8 }}
                rowGap={{ base: 4, md: 6, lg: 8 }}
              >
                {menuItems.map((item) => (
                  <MenuCard
                    key={item.title}
                    item={item}
                    onClick={() => handleNavigate(item.path, item.external)}
                  />
                ))}
              </SimpleGrid>
            </Stack>
          </Stack>
        </Container>
      </Box>
    </Box>
  )
}
