import React, { useState, useEffect } from 'react';
import { ChakraProvider, Box, VStack, Container } from '@chakra-ui/react';
import EEGDisplay from './components/EEGDisplay';
import ChatWindow from './components/ChatWindow';
import EmotionDisplay from './components/EmotionDisplay';

function App() {
  const [emotion, setEmotion] = useState(null);
  const [eegData, setEEGData] = useState([]);
  
  // Poll emotion data
  useEffect(() => {
    const pollEmotion = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/emotion');
        const data = await response.json();
        setEmotion(data.emotion);
        setEEGData(data.eeg_data);
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
        <VStack spacing={5}>
          <Box w="100%">
            <EEGDisplay data={eegData} />
          </Box>
          <Box w="100%">
            <EmotionDisplay emotion={emotion} />
          </Box>
          <Box w="100%">
            <ChatWindow currentEmotion={emotion} />
          </Box>
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App; 