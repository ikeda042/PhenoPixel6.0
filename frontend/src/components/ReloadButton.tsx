import { useMemo, useState } from 'react'
import { Button, HStack, Icon, Text } from '@chakra-ui/react'
import { RotateCw } from 'lucide-react'
import { getApiBase } from '../utils/apiBase'

type GitPullResponse = {
  status?: string
  output?: string
  detail?: string
}

const ReloadButton = () => {
  const apiBase = useMemo(() => getApiBase(), [])
  const [isUpdating, setIsUpdating] = useState(false)

  const handleUpdate = async () => {
    if (isUpdating) return
    setIsUpdating(true)
    try {
      const response = await fetch(`${apiBase}/system/git-pull`, { method: 'POST' })
      const payload = (await response.json().catch(() => ({}))) as GitPullResponse
      if (!response.ok) {
        const message =
          typeof payload.detail === 'string' && payload.detail.trim()
            ? payload.detail
            : 'Update failed.'
        throw new Error(message)
      }
      if (typeof payload.output === 'string' && payload.output.trim()) {
        console.info(payload.output)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error'
      console.error('Update failed:', error)
      window.alert(`Update failed: ${message}`)
    } finally {
      setIsUpdating(false)
    }
  }

  return (
    <Button
      type="button"
      size="sm"
      border="1px solid"
      borderColor="tide.500"
      bg="tide.500"
      color="ink.900"
      _hover={{ bg: 'tide.400' }}
      onClick={handleUpdate}
      loading={isUpdating}
      loadingText="Updating"
      aria-label="Update application"
    >
      <HStack spacing="2">
        <Icon as={RotateCw} boxSize={4} />
        <Text
          fontSize="xs"
          letterSpacing="0.12em"
          textTransform="uppercase"
          display={{ base: 'none', md: 'inline' }}
        >
          Update
        </Text>
      </HStack>
    </Button>
  )
}

export default ReloadButton
