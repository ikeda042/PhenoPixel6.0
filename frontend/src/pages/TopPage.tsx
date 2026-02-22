import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Badge,
  Box,
  Container,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Icon,
  SimpleGrid,
  Stack,
  Text,
  VStack,
} from '@chakra-ui/react'
import {
  Activity,
  ChevronRight,
  Cpu,
  Database,
  Folder,
  Server,
  Share2,
  Wifi,
  WifiOff,
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
    ok: { bg: 'sand.100', color: 'tide.300', dot: 'tide.300' },
    error: { bg: 'sand.100', color: 'violet.400', dot: 'violet.400' },
    unknown: { bg: 'sand.100', color: 'ink.700', dot: 'sand.300' },
  } as const
  const style = palette[tone]

  return (
    <HStack spacing="2" px="2.5" py="1" borderRadius="md" bg={style.bg} border="1px solid" borderColor="sand.200">
      <Box w="2px" h="12px" borderRadius="full" bg={style.dot} />
      <Icon as={icon} boxSize={3.5} color={style.color} />
      <Text fontSize="xs" fontWeight="500" color="ink.700">
        {label}:
      </Text>
      <Text fontSize="xs" fontWeight="700" color={style.color} textTransform="uppercase">
        {value}
      </Text>
    </HStack>
  )
}

const MenuListItem = ({ item, onClick }: { item: MenuItem; onClick: () => void }) => (
  <Flex
    as="button"
    w="full"
    onClick={onClick}
    align="center"
    justify="space-between"
    p="3"
    bg="sand.100"
    border="1px solid"
    borderColor="sand.200"
    borderRadius="md"
    transition="all 0.2s"
    _hover={{
      borderColor: 'tide.400',
      bg: 'sand.200',
      transform: 'translateX(2px)',
    }}
    group
  >
    <HStack spacing="3">
      <Flex
        w="8"
        h="8"
        borderRadius="md"
        bg="sand.50"
        align="center"
        justify="center"
        color="ink.700"
        border="1px solid"
        borderColor="sand.200"
      >
        <Icon as={item.icon} boxSize={4} />
      </Flex>
      <Box textAlign="left">
        <Text fontSize="sm" fontWeight="600" color="ink.900">
          {item.title}
        </Text>
        <Text fontSize="xs" color="ink.700" noOfLines={1}>
          {item.description}
        </Text>
      </Box>
    </HStack>
    <Icon as={ChevronRight} boxSize={4} color="ink.700" />
  </Flex>
)

const menuItems: MenuItem[] = [
  {
    title: 'Cell Extraction',
    description: 'Extract cells from ND2 microscopy files.',
    path: '/nd2files',
    icon: Cpu,
  },
  {
    title: 'Database Console',
    description: 'Label cells and manage datasets.',
    path: '/databases',
    icon: Database,
  },
  {
    title: 'File Manager',
    description: 'Manage files on the local server.',
    path: '/files',
    icon: Folder,
  },
  {
    title: 'Graph Engine',
    description: 'Generate graph metrics and plots from CSV inputs.',
    path: '/graph-engine',
    icon: Share2,
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
        if (!res.ok) throw new Error('Failed to load activity')
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

  const activityTotal = useMemo(() => activityPoints.reduce((sum, point) => sum + point.count, 0), [activityPoints])
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
    return `${formatLongDate(activityPoints[0].date)} - ${formatLongDate(activityPoints[activityPoints.length - 1].date)}`
  }, [activityPoints])
  const activityLabels = useMemo(() => activityPoints.map((point) => formatShortDate(point.date)), [activityPoints])

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
    const areaPath = lineSegment ? `M ${firstX} ${baselineY} L ${lineSegment} L ${lastX} ${baselineY} Z` : ''
    const gridLines = [0, 0.5, 1].map((ratio) => ({ y: paddingY + innerHeight - ratio * innerHeight }))

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
    <Box minH="100vh" bg="sand.50" color="ink.900">
      <PageHeader
        actions={
          <HStack spacing="2">
            <ReloadButton />
            <ThemeToggleButton />
          </HStack>
        }
      />

      <Container maxW="container.xl" py={6}>
        <Flex justify="space-between" align="flex-end" mb="6">
          <Box>
            <Heading size="lg" color="ink.900" fontWeight="600">
              Overview
            </Heading>
          </Box>
          <HStack spacing="3" display={{ base: 'none', md: 'flex' }}>
            <StatusChip label="Backend" value={backendStatus ?? 'Checking'} tone={backendTone} icon={Server} />
            <StatusChip
              label="Network"
              value={internetStatus === null ? 'Checking' : internetStatus ? 'Online' : 'Offline'}
              tone={internetTone}
              icon={internetStatus ? Wifi : WifiOff}
            />
          </HStack>
        </Flex>

        <HStack spacing="3" display={{ base: 'flex', md: 'none' }} mb="6">
          <StatusChip label="Backend" value={backendStatus ?? 'Checking'} tone={backendTone} icon={Server} />
          <StatusChip
            label="Network"
            value={internetStatus === null ? 'Checking' : internetStatus ? 'Online' : 'Offline'}
            tone={internetTone}
            icon={internetStatus ? Wifi : WifiOff}
          />
        </HStack>

        <Grid templateColumns={{ base: '1fr', lg: '3fr 1fr' }} gap="6">
          <GridItem>
            <Stack spacing="6">
              <SimpleGrid columns={{ base: 1, md: 3 }} gap="4">
                {[
                  { label: 'Total Actions', value: activityTotal, borderTop: 'tide.500' },
                  { label: 'Daily Average', value: activityAverage, borderTop: 'tide.300' },
                  {
                    label: 'Peak Usage',
                    value: activityPeak ? activityPeak.count : 0,
                    subtext: activityPeak ? formatLongDate(activityPeak.date) : '-',
                    borderTop: 'violet.400',
                  },
                ].map((stat, i) => (
                  <Box key={i} bg="sand.100" p="4" borderRadius="md" border="1px solid" borderColor="sand.200" borderTopWidth="3px" borderTopColor={stat.borderTop} shadow="sm">
                    <Text fontSize="xs" color="ink.700" fontWeight="600" textTransform="uppercase" letterSpacing="wider" mb="1">
                      {stat.label}
                    </Text>
                    <HStack align="baseline" justify="space-between">
                      <Text fontSize="2xl" fontWeight="700" color="ink.900" lineHeight="1">
                        {stat.value}
                      </Text>
                      {stat.subtext && (
                        <Text fontSize="xs" color="ink.700" fontWeight="500">
                          {stat.subtext}
                        </Text>
                      )}
                    </HStack>
                  </Box>
                ))}
              </SimpleGrid>

              <Box bg="sand.100" border="1px solid" borderColor="sand.200" borderRadius="md" shadow="sm" p="5">
                <HStack justify="space-between" mb="6">
                  <HStack spacing="2">
                    <Icon as={Activity} boxSize={5} color="tide.400" />
                    <Heading size="sm" fontWeight="600" color="ink.900">
                      Weekly Activity Trends
                    </Heading>
                  </HStack>
                  <Badge bg="sand.200" color="ink.900" fontSize="xs" px="2" py="1" borderRadius="md">
                    {activityRangeLabel}
                  </Badge>
                </HStack>

                <Box h="240px" w="full" mt="4">
                  {activityStatus === 'loading' || activityStatus === 'idle' ? (
                    <Flex justify="center" align="center" h="full">
                      <Text fontSize="sm" color="ink.700">Loading metrics...</Text>
                    </Flex>
                  ) : activityStatus === 'error' ? (
                    <Flex justify="center" align="center" h="full">
                      <Text fontSize="sm" color="violet.400">Activity data unavailable.</Text>
                    </Flex>
                  ) : activityChart ? (
                    <Box w="full" h="200px" position="relative">
                      <svg
                        width="100%"
                        height="100%"
                        viewBox={`0 0 ${activityChart.width} ${activityChart.height}`}
                        preserveAspectRatio="none"
                        style={{ overflow: 'visible' }}
                      >
                        <defs>
                          <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="var(--chakra-colors-tide-400)" stopOpacity={0.2} />
                            <stop offset="100%" stopColor="var(--chakra-colors-tide-400)" stopOpacity={0} />
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
                            strokeWidth="1"
                          />
                        ))}
                        {activityChart.areaPath && <path d={activityChart.areaPath} fill="url(#chartGradient)" />}
                        {activityChart.linePath && (
                          <path
                            d={activityChart.linePath}
                            fill="none"
                            stroke="var(--chakra-colors-tide-400)"
                            strokeWidth="2.5"
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
                              w="10px"
                              h="10px"
                              bg="sand.50"
                              border="2px solid"
                              borderColor="tide.400"
                              borderRadius="full"
                              boxShadow="sm"
                              zIndex={2}
                            />
                            <Text
                              position="absolute"
                              left={`${leftPercent}%`}
                              bottom="-24px"
                              transform="translateX(-50%)"
                              fontSize="xs"
                              fontWeight="500"
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
                      <Text fontSize="sm" color="ink.700">No activity data yet.</Text>
                    </Flex>
                  )}
                </Box>
              </Box>
            </Stack>
          </GridItem>

          <GridItem>
            <Box bg="sand.100" border="1px solid" borderColor="sand.200" borderRadius="md" shadow="sm" p="5">
              <Heading size="sm" fontWeight="600" color="ink.900" mb="4">
                System Modules
              </Heading>
              <VStack spacing="2" align="stretch">
                {menuItems.map((item) => (
                  <MenuListItem
                    key={item.title}
                    item={item}
                    onClick={() => handleNavigate(item.path, item.external)}
                  />
                ))}
              </VStack>
            </Box>
          </GridItem>
        </Grid>
      </Container>
    </Box>
  )
}
