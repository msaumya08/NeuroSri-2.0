import React, { useState, useEffect } from 'react';
import { ChakraProvider, Box, VStack, Container, useColorMode, Switch, HStack, Text, Flex } from '@chakra-ui/react';
import EEGDisplay from './components/EEGDisplay';
import ChatWindow from './components/ChatWindow';
import EmotionDisplay from './components/EmotionDisplay';
import { api } from './services/api';

function ColorModeToggle() {
  const { colorMode, toggleColorMode } = useColorMode();
  return (
    <HStack justifyContent="flex-end" w="100%" mb={2}>
      <Text fontSize="sm">{colorMode === 'light' ? 'Light' : 'Dark'} Mode</Text>
      <Switch isChecked={colorMode === 'dark'} onChange={toggleColorMode} />
    </HStack>
  );
}

function App() {
  const [emotionData, setEmotionData] = useState({
    emotion: null,
    confidence: null,
    chat_message: null,
    is_setup_phase: true,
    setup_complete: false
  });
  const [eegData, setEEGData] = useState([]);
  
  // Poll emotion data
  useEffect(() => {
    const pollEmotion = async () => {
      try {
        const data = await api.getEmotion();
        setEmotionData(data);
        setEEGData(data.eeg_data || []);
      } catch (error) {
        console.error('Error fetching emotion:', error);
      }
    };

    const interval = setInterval(pollEmotion, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ChakraProvider>
      <Container maxW="container.xl" py={5}>
        <ColorModeToggle />
        <Flex direction={{ base: 'column', md: 'row' }} gap={5} w="100%">
          <Box w={{ base: '100%', md: '60%' }}>
            <EEGDisplay data={eegData} />
          </Box>
          <Box w={{ base: '100%', md: '40%' }}>
            <ChatWindow 
              currentEmotion={emotionData.emotion} 
              emotionData={emotionData}
            />
          </Box>
        </Flex>
      </Container>
    </ChakraProvider>
  );
}

export default App;