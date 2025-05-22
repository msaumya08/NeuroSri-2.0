import React, { useState } from 'react';
import {
  Box,
  VStack,
  Input,
  Button,
  Text,
  useToast
} from '@chakra-ui/react';

function ChatWindow({ currentEmotion }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const toast = useToast();

  const sendMessage = async () => {
    if (!input.trim()) return;

    try {
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: input,
          emotion: currentEmotion
        }),
      });

      const data = await response.json();
      
      setMessages([
        ...messages,
        { text: input, sender: 'user' },
        { text: data.response, sender: 'bot' }
      ]);
      
      setInput('');
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to send message',
        status: 'error',
        duration: 3000,
      });
    }
  };

  return (
    <Box borderWidth={1} borderRadius="lg" p={4}>
      <VStack spacing={4}>
        <Box h="400px" overflowY="auto" w="100%">
          {messages.map((msg, idx) => (
            <Text 
              key={idx}
              alignSelf={msg.sender === 'user' ? 'flex-end' : 'flex-start'}
              bg={msg.sender === 'user' ? 'blue.100' : 'gray.100'}
              p={2}
              borderRadius="md"
              my={1}
            >
              {msg.text}
            </Text>
          ))}
        </Box>
        <Box w="100%" display="flex">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type a message..."
            mr={2}
          />
          <Button onClick={sendMessage}>Send</Button>
        </Box>
      </VStack>
    </Box>
  );
}

export default ChatWindow; 