import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { Link as RouterLink, useNavigate } from 'react-router-dom'
import {
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
  Icon,
  Input,
  InputGroup,
  Stack,
  Text,
} from '@chakra-ui/react'
import AppHeader from '../components/AppHeader'
import ReloadButton from '../components/ReloadButton'
import ThemeToggle from '../components/ThemeToggle'
import { Search, Trash2 } from 'lucide-react'
import { getApiBase } from '../utils/apiBase'

type Nd2FilesResponse = {
  files?: string[]
}

type Nd2BulkDeleteResponse = {
  deleted?: string[]
  invalid?: string[]
  missing?: string[]
}

const SearchGlyph = () => (
  <Box position="relative" w="16px" h="16px" color="ink.700">
    <Search size={16} />
  </Box>
)

export default function Nd2FilesPage() {
  const navigate = useNavigate()
  const apiBase = useMemo(() => getApiBase(), [])
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [files, setFiles] = useState<string[]>([])
  const [searchText, setSearchText] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set())
  const [isLoading, setIsLoading] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [isBulkDeleting, setIsBulkDeleting] = useState(false)
  const [deletingFile, setDeletingFile] = useState<string | null>(null)
  const [parsingFile, setParsingFile] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchFiles = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/nd2_files`, {
        headers: { accept: 'application/json' },
      })
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`)
      }
      const data = (await res.json()) as Nd2FilesResponse
      setFiles(Array.isArray(data.files) ? data.files : [])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load files')
      setFiles([])
    } finally {
      setIsLoading(false)
    }
  }, [apiBase])

  const uploadFile = useCallback(
    async (file: File) => {
      setIsUploading(true)
      setError(null)
      try {
        const formData = new FormData()
        formData.append('file', file)
        const res = await fetch(`${apiBase}/nd2_files`, {
          method: 'POST',
          headers: { accept: 'application/json' },
          body: formData,
        })
        if (!res.ok) {
          throw new Error(`Upload failed (${res.status})`)
        }
        await res.json()
        await fetchFiles()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to upload file')
      } finally {
        setIsUploading(false)
      }
    },
    [apiBase, fetchFiles],
  )

  const handleDelete = useCallback(
    async (file: string) => {
      if (deletingFile || isBulkDeleting) return
      const confirmed = window.confirm(`Delete ${file}?`)
      if (!confirmed) return
      setDeletingFile(file)
      setError(null)
      try {
        const res = await fetch(
          `${apiBase}/nd2_files/${encodeURIComponent(file)}`,
          { method: 'DELETE', headers: { accept: 'application/json' } },
        )
        if (!res.ok) {
          throw new Error(`Delete failed (${res.status})`)
        }
        await res.json()
        await fetchFiles()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete file')
      } finally {
        setDeletingFile(null)
      }
    },
    [apiBase, deletingFile, fetchFiles, isBulkDeleting],
  )

  const handleParse = useCallback(
    async (file: string) => {
      if (parsingFile) return
      setParsingFile(file)
      setError(null)
      try {
        const res = await fetch(`${apiBase}/nd2parser/parse`, {
          method: 'POST',
          headers: {
            accept: 'application/json',
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ nd2file: file }),
        })
        if (!res.ok) {
          throw new Error(`Parse failed (${res.status})`)
        }
        await res.json()
        navigate(`/nd2parser?nd2file=${encodeURIComponent(file)}`)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to parse file')
      } finally {
        setParsingFile(null)
      }
    },
    [apiBase, navigate, parsingFile],
  )

  useEffect(() => {
    void fetchFiles()
  }, [fetchFiles])

  useEffect(() => {
    setSelectedFiles((prev) => {
      if (prev.size === 0) return prev
      const available = new Set(files)
      let changed = false
      const next = new Set<string>()
      prev.forEach((file) => {
        if (available.has(file)) {
          next.add(file)
        } else {
          changed = true
        }
      })
      return changed ? next : prev
    })
  }, [files])

  const handleUploadClick = () => {
    inputRef.current?.click()
  }

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    void uploadFile(file)
    event.target.value = ''
  }

  const filteredFiles = useMemo(() => {
    const query = searchText.trim().toLowerCase()
    if (!query) return files
    return files.filter((file) => file.toLowerCase().includes(query))
  }, [files, searchText])

  const selectedCount = selectedFiles.size
  const filteredSelectedCount = useMemo(() => {
    if (filteredFiles.length === 0 || selectedFiles.size === 0) return 0
    let count = 0
    filteredFiles.forEach((file) => {
      if (selectedFiles.has(file)) {
        count += 1
      }
    })
    return count
  }, [filteredFiles, selectedFiles])
  const allFilteredSelected =
    filteredFiles.length > 0 && filteredSelectedCount === filteredFiles.length

  const handleSelectionChange = useCallback((file: string, checked: boolean) => {
    setSelectedFiles((prev) => {
      const next = new Set(prev)
      if (checked) {
        next.add(file)
      } else {
        next.delete(file)
      }
      return next
    })
  }, [])

  const handleSelectAll = useCallback(
    (checked: boolean) => {
      if (filteredFiles.length === 0) return
      setSelectedFiles((prev) => {
        const next = new Set(prev)
        filteredFiles.forEach((file) => {
          if (checked) {
            next.add(file)
          } else {
            next.delete(file)
          }
        })
        return next
      })
    },
    [filteredFiles],
  )

  const handleBulkDelete = useCallback(async () => {
    if (isBulkDeleting || deletingFile || selectedFiles.size === 0) return
    const selectedList = Array.from(selectedFiles)
    const confirmed = window.confirm(`Delete ${selectedList.length} file(s)?`)
    if (!confirmed) return
    setIsBulkDeleting(true)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/nd2_files/bulk-delete`, {
        method: 'POST',
        headers: {
          accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filenames: selectedList }),
      })
      if (!res.ok) {
        throw new Error(`Bulk delete failed (${res.status})`)
      }
      const data = (await res.json()) as Nd2BulkDeleteResponse
      await fetchFiles()
      setSelectedFiles(new Set())
      const issues: string[] = []
      if (data.missing?.length) {
        issues.push(`Missing: ${data.missing.join(', ')}`)
      }
      if (data.invalid?.length) {
        issues.push(`Invalid: ${data.invalid.join(', ')}`)
      }
      if (issues.length > 0) {
        setError(`Some files could not be deleted. ${issues.join(' ')}`)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete files')
    } finally {
      setIsBulkDeleting(false)
    }
  }, [apiBase, deletingFile, fetchFiles, isBulkDeleting, selectedFiles])

  return (
    <Box minH="100vh" bg="sand.50" color="ink.900">
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
            Extractor
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
                <BreadcrumbCurrentLink color="ink.900">ND2 Files</BreadcrumbCurrentLink>
              </BreadcrumbItem>
            </BreadcrumbList>
          </BreadcrumbRoot>
          <ReloadButton />
          <ThemeToggle />
        </HStack>
      </AppHeader>

      <Container maxW="72.5rem" py={{ base: 8, md: 12 }}>
        <Stack spacing="6">
          <Stack
            direction={{ base: 'column', md: 'row' }}
            align={{ base: 'stretch', md: 'center' }}
            justify="space-between"
            gap="3"
          >
            <InputGroup
              size="sm"
              maxW={{ base: '100%', md: '360px' }}
              startElement={<SearchGlyph />}
              bg="sand.100"
              borderRadius="md"
            >
              <Input
                placeholder="Search ND2 files"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
                border="1px solid"
                borderColor="sand.200"
                color="ink.900"
                _placeholder={{ color: 'ink.700' }}
                _focusVisible={{
                  borderColor: 'tide.400',
                  boxShadow: '0 0 0 1px rgba(45,212,191,0.6)',
                }}
              />
            </InputGroup>

            <Button
              size="sm"
              bg="tide.500"
              color="ink.900"
              _hover={{ bg: 'tide.400' }}
              onClick={handleUploadClick}
              loading={isUploading}
              alignSelf={{ base: 'flex-start', md: 'center' }}
            >
              Upload ND2
            </Button>
          </Stack>

          <HStack
            spacing="3"
            align="center"
            justify="space-between"
            flexWrap="wrap"
          >
            <HStack spacing="3" align="center" flexWrap="wrap">
              <Checkbox.Root
                checked={allFilteredSelected}
                onCheckedChange={(details) =>
                  handleSelectAll(details.checked === true)
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
                  Select all
                </Checkbox.Label>
              </Checkbox.Root>
              <Text fontSize="xs" color="ink.700">
                Selected: {selectedCount}
              </Text>
            </HStack>
            <HStack spacing="2" align="center" flexWrap="wrap">
              <Button
                size="xs"
                variant="outline"
                borderColor="sand.200"
                color="ink.700"
                _hover={{ bg: 'sand.200', color: 'ink.900' }}
                onClick={() => setSelectedFiles(new Set())}
                isDisabled={selectedCount === 0}
              >
                Clear
              </Button>
              <Button
                size="xs"
                variant="outline"
                borderColor="red.400"
                color="red.400"
                _hover={{ bg: 'red.500/10' }}
                onClick={handleBulkDelete}
                loading={isBulkDeleting}
                isDisabled={selectedCount === 0 || isBulkDeleting || !!deletingFile}
              >
                Delete selected
              </Button>
            </HStack>
          </HStack>

          <Box
            bg="sand.100"
            border="1px solid"
            borderColor="sand.200"
            borderRadius="xl"
            overflow="hidden"
          >
            {isLoading && (
              <Box px="4" py="5">
                <Text fontSize="sm" color="ink.700">
                  Loading files...
                </Text>
              </Box>
            )}

            {!isLoading && error && (
              <Box px="4" py="5">
                <Text fontSize="sm" color="violet.300">
                  {error}
                </Text>
              </Box>
            )}

            {!isLoading && !error && files.length === 0 && (
              <Box px="4" py="5">
                <Text fontSize="sm" color="ink.700">
                  No ND2 files found.
                </Text>
              </Box>
            )}

            {!isLoading &&
              !error &&
              files.length > 0 &&
              filteredFiles.length === 0 && (
                <Box px="4" py="5">
                  <Text fontSize="sm" color="ink.700">
                    No matching files.
                  </Text>
                </Box>
              )}

            {!isLoading &&
              !error &&
              filteredFiles.map((file, index) => (
                <Grid
                  key={file}
                  templateColumns={{
                    base: 'minmax(0, 1fr)',
                    md: 'minmax(0, 1fr) 18rem',
                  }}
                  px="4"
                  py="3"
                  borderBottom={
                    index === filteredFiles.length - 1 ? 'none' : '1px solid'
                  }
                  borderColor="sand.200"
                  _hover={{ bg: 'sand.200' }}
                  transition="background 0.2s ease"
                >
                  <Checkbox.Root
                    checked={selectedFiles.has(file)}
                    onCheckedChange={(details) =>
                      handleSelectionChange(file, details.checked === true)
                    }
                    colorPalette="tide"
                    display="flex"
                    alignItems="center"
                    gap="3"
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
                    <Checkbox.Label fontSize="sm" fontWeight="500" color="ink.900">
                      {file}
                    </Checkbox.Label>
                  </Checkbox.Root>
                  <HStack
                    justify={{ base: 'flex-start', md: 'flex-end' }}
                    spacing="2"
                    flexWrap={{ base: 'wrap', md: 'nowrap' }}
                  >
                    <Button
                      size="xs"
                      bg="tide.500"
                      color="ink.900"
                      _hover={{ bg: 'tide.400' }}
                      onClick={() =>
                        navigate(`/cell-extraction?filename=${encodeURIComponent(file)}`)
                      }
                    >
                      Extract cells
                    </Button>
                    <Button
                      size="xs"
                      bg="tide.500"
                      color="ink.900"
                      _hover={{ bg: 'tide.400' }}
                      onClick={() => handleParse(file)}
                      loading={parsingFile === file}
                      isDisabled={isBulkDeleting}
                    >
                      ND2 viewer
                    </Button>
                    <Button
                      size="xs"
                      variant="outline"
                      borderColor="red.400"
                      color="red.400"
                      _hover={{ bg: 'red.500/10' }}
                      onClick={() => handleDelete(file)}
                      loading={deletingFile === file}
                      isDisabled={isBulkDeleting}
                      aria-label={`Delete ${file}`}
                      minW="auto"
                      px="2"
                    >
                      <Icon as={Trash2} boxSize={3.5} />
                    </Button>
                  </HStack>
                </Grid>
              ))}
          </Box>
        </Stack>
      </Container>

      <input
        ref={inputRef}
        type="file"
        accept=".nd2"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
    </Box>
  )
}
