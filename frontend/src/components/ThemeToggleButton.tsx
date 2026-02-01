import { useState } from 'react'
import { Button, HStack, Icon, Text } from '@chakra-ui/react'
import { Moon, Sun } from 'lucide-react'

const THEME_STORAGE_KEY = 'phenopixel-theme'

type ThemeMode = 'dark' | 'light'

const getCurrentMode = (): ThemeMode => {
  if (typeof document === 'undefined') return 'dark'
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

const ThemeToggleButton = () => {
  const [mode, setMode] = useState<ThemeMode>(() => getCurrentMode())

  const handleToggle = () => {
    const nextMode: ThemeMode = mode === 'dark' ? 'light' : 'dark'
    const root = document.documentElement
    root.classList.toggle('dark', nextMode === 'dark')
    root.classList.toggle('light', nextMode === 'light')
    window.localStorage.setItem(THEME_STORAGE_KEY, nextMode)
    setMode(nextMode)
  }

  const targetMode: ThemeMode = mode === 'dark' ? 'light' : 'dark'
  const label = targetMode === 'dark' ? 'Dark' : 'Light'
  const ModeIcon = targetMode === 'dark' ? Moon : Sun

  return (
    <Button
      type="button"
      size="sm"
      border="1px solid"
      borderColor="tide.500"
      bg="transparent"
      color="tide.500"
      _hover={{ bg: 'tide.500', color: 'ink.900' }}
      onClick={handleToggle}
      aria-label={`Switch to ${label} mode`}
    >
      <HStack spacing="2">
        <Icon as={ModeIcon} boxSize={4} />
        <Text
          fontSize="xs"
          letterSpacing="0.12em"
          textTransform="uppercase"
          display={{ base: 'none', md: 'inline' }}
        >
          {label}
        </Text>
      </HStack>
    </Button>
  )
}

export default ThemeToggleButton
