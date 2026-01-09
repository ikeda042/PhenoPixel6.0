import type { ReactNode } from 'react'
import { Flex } from '@chakra-ui/react'
import type { FlexProps } from '@chakra-ui/react'

type AppHeaderProps = Omit<FlexProps, 'children'> & {
  children: ReactNode
  bg?: FlexProps['bg']
}

const AppHeader = ({ children, bg = 'sand.50/85', ...rest }: AppHeaderProps) => (
  <Flex
    as="header"
    align="center"
    justify="space-between"
    px={{ base: 4, md: 8 }}
    h="4rem"
    borderBottom="1px solid"
    borderColor="sand.200"
    bg={bg}
    backdropFilter="blur(0.75rem)"
    position="sticky"
    top="0"
    zIndex="sticky"
    {...rest}
  >
    {children}
  </Flex>
)

export default AppHeader
