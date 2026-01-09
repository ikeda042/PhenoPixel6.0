import { useCallback, useEffect, useMemo, useState } from 'react'
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
