import React from 'react';
import { Box, Text, Progress, VStack, HStack, Badge } from '@chakra-ui/react';

function EmotionDisplay({ emotion, confidence }) {
    // Calculate percentages for both emotions
    const relaxedPercentage = emotion === 'relaxed' ? confidence * 100 : (1 - confidence) * 100;
    const stressedPercentage = 100 - relaxedPercentage;
    
    // Determine dominant emotion and its percentage
    const dominantEmotion = relaxedPercentage >= stressedPercentage ? 'RELAXED' : 'STRESSED';
    
    return (
        <Box borderWidth={1} borderRadius="lg" p={4} bg="white">
            <VStack spacing={4} align="stretch">
                <Text fontSize="xl" fontWeight="bold" textAlign="center">
                    Current Emotional State
                </Text>
                
                <Box textAlign="center">
                    <Text fontSize="2xl" color={dominantEmotion === 'RELAXED' ? 'green.400' : 'red.400'} fontWeight="bold">
                        {dominantEmotion}
                    </Text>
                    <HStack spacing={4} justify="center" mt={2}>
                        <Badge colorScheme="green" variant="subtle" fontSize="md" px={3} py={1}>
                            Relaxed: {relaxedPercentage.toFixed(1)}%
                        </Badge>
                        <Badge colorScheme="red" variant="subtle" fontSize="md" px={3} py={1}>
                            Stressed: {stressedPercentage.toFixed(1)}%
                        </Badge>
                    </HStack>
                </Box>

                <Box position="relative" pt={2}>
                    <Text fontSize="sm" color="gray.600" mb={2}>Emotion Distribution</Text>
                    <Box 
                        position="relative" 
                        height="24px"
                        borderRadius="full"
                        overflow="hidden"
                    >
                        {/* Relaxed portion */}
                        <Box
                            position="absolute"
                            left="0"
                            height="100%"
                            width={`${relaxedPercentage}%`}
                            bg="green.300"
                            transition="width 0.3s ease"
                        />
                        {/* Stressed portion */}
                        <Box
                            position="absolute"
                            right="0"
                            height="100%"
                            width={`${stressedPercentage}%`}
                            bg="red.300"
                            transition="width 0.3s ease"
                        />
                        
                        {/* Percentage labels */}
                        <HStack
                            position="absolute"
                            width="100%"
                            height="100%"
                            justify="space-between"
                            px={2}
                            color="white"
                            fontWeight="bold"
                        >
                            <Text fontSize="sm" textShadow="0px 0px 3px rgba(0,0,0,0.3)">
                                {relaxedPercentage.toFixed(1)}%
                            </Text>
                            <Text fontSize="sm" textShadow="0px 0px 3px rgba(0,0,0,0.3)">
                                {stressedPercentage.toFixed(1)}%
                            </Text>
                        </HStack>
                    </Box>
                    
                    {/* Legend */}
                    <HStack justify="space-between" mt={1}>
                        <Text fontSize="xs" color="green.500">Relaxed</Text>
                        <Text fontSize="xs" color="red.500">Stressed</Text>
                    </HStack>
                </Box>
            </VStack>
        </Box>
    );
}

export default EmotionDisplay;