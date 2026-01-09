import { useEffect, useState } from 'react'
import { Button, HStack, Icon, Text } from '@chakra-ui/react'
import { Moon, Sun } from 'lucide-react'
import { applyThemeMode, getInitialTheme } from '../utils/themeMode'
import type { ThemeMode } from '../utils/themeMode'

const ThemeToggle = () => {
  const [mode, setMode] = useState<ThemeMode>(() => getInitialTheme())
  const isDark = mode === 'dark'

  useEffect(() => {
    applyThemeMode(mode)
  }, [mode])

  return (
    <Button
      type="button"
      size="sm"
      border="1px solid"
      borderColor="sand.200"
      bg="sand.100"
      color="ink.900"
      _hover={{ bg: 'sand.200' }}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      onClick={() => setMode(isDark ? 'light' : 'dark')}
    >
      <HStack spacing="2">
        <Icon as={isDark ? Sun : Moon} boxSize={4} />
        <Text
          fontSize="xs"
          letterSpacing="0.12em"
          textTransform="uppercase"
          display={{ base: 'none', md: 'inline' }}
        >
          {isDark ? 'Light' : 'Dark'}
        </Text>
      </HStack>
    </Button>
  )
}

export default ThemeToggle
