import React, { useState, useEffect, useCallback, useRef, memo } from 'react';
import {
    Box,
    VStack,
    Input,
    Button,
    Text,
    useToast,
    Flex,
    Avatar,
    HStack,
    IconButton,
    Tooltip,
    Select,
    FormControl,
    FormLabel,
    NumberInput,
    NumberInputField,
    NumberInputStepper,
    NumberIncrementStepper,
    NumberDecrementStepper,
    Textarea,
    Heading,
    Radio,
    RadioGroup,
    Stack,
    Divider,
    useColorMode
} from '@chakra-ui/react';
import { FaMicrophone, FaStop, FaLanguage, FaVolumeMute, FaUser, FaPaperPlane } from 'react-icons/fa';
import { api } from '../services/api';
import { keyframes } from '@emotion/react';

// Avatar paths for the profile pictures - images should be in the public/images folder
const BOT_AVATAR = "/images/NeuroSri_logo.png";  // Chatbot's profile picture
const USER_AVATAR = "/images/User_logo.png";     // User's profile picture

// Language options with enhanced female voice settings
const LANGUAGES = {
    ENGLISH: {
        code: 'en-US',
        name: 'English',
        rate: 1.30,     // Increased speed for faster speech (was 0.92)
        pitch: 1.0,    // Higher pitch for distinctly female voice
        voiceName: null // Will be set dynamically
    },
    HINDI: {
        code: 'hi-IN',
        name: 'Hindi',
        rate: 1.30,     // Increased speed for faster speech (was 0.92)
        pitch: 1.0,    // Higher pitch for distinctly female voice
        voiceName: null // Will be set dynamically
    }
};

// Speech synthesis configuration - Enhanced for more realistic female voice
const VOICE_VOLUME = 1.0;  // Maximum volume

// Keep track of speech synthesis state
let speechQueue = [];
let isSpeaking = false;

// Punctuation timings for more natural speech
const PUNCTUATION_PAUSES = {
    ',': 50,
    '.': 150,
    '!': 200,
    '?': 200,
    ';': 250,
    ':': 200,
    '—': 400,
    '–': 400,
    '*': 0
};

// Global function to ensure speech is fully stopped
const ensureSpeechStopped = () => {
    // Cancel any active speech
    speechSynthesis.cancel();
    
    // Clear our internal queue tracking
    speechQueue = [];
    isSpeaking = false;
    
    // Double check that speech synthesis is really stopped
    if (speechSynthesis.speaking || speechSynthesis.pending) {
        speechSynthesis.cancel();
    }
};

// Function to process speech queue
const processSpeechQueue = () => {
    if (speechQueue.length === 0 || isSpeaking) return;
    
    isSpeaking = true;
    const utterance = speechQueue[0];
    
    try {
    speechSynthesis.speak(utterance);
    } catch (error) {
        console.error('Error in speech synthesis:', error);
        // Move to the next item if this one fails
        speechQueue.shift();
        isSpeaking = false;
        processSpeechQueue();
    }
};

// Expanded list of terms that indicate a female voice
const FEMALE_IDENTIFIERS = [
    'female', 'woman', 'girl', 'feminine', 'lady', 'women', 
    'f ', ' f)', '(f)', 'female)', '(female', 'fem)', '(fem'
];

// Expanded list of common female names for voice detection
const FEMALE_NAMES = [
    'zira', 'samantha', 'siri', 'alexa', 'cortana', 'karen', 'susan', 
    'lisa', 'amy', 'emma', 'olivia', 'emily', 'sarah', 'michelle', 
    'hazel', 'moira', 'tessa', 'fiona', 'veena', 'victoria', 'elizabeth', 
    'catherine', 'kathy', 'katherine', 'kate', 'allison', 'ava', 'joanna',
    'sophia', 'nicole', 'jennifer', 'julie', 'melanie', 'alessa', 'rosa',
    'kalpana', 'heera', 'priya', 'neerja', 'swara', 'dipti', 'lekha'
];

// Function to evaluate voice quality - heavily prioritize female voices
const evaluateVoiceQuality = (voice, languageCode) => {
    let score = 0;
    
    // Make sure the voice matches our target language code
    if (!voice.lang.includes(languageCode.substring(0, 2))) {
        return -1; // Not the right language
    }
    
    // Check if it's explicitly a female voice
    const voiceNameLower = voice.name.toLowerCase();
    
    // MAJOR boost for voices that explicitly identify as female
    if (FEMALE_IDENTIFIERS.some(term => voiceNameLower.includes(term))) {
        score += 50; // Heavily prioritize explicitly female voices
    }
    
    // Decent boost for voices with female names
    if (FEMALE_NAMES.some(name => voiceNameLower.includes(name))) {
        score += 30;
    }
    
    // Prefer voices that are marked as "premium" or "enhanced" in their names
    if (voiceNameLower.includes('premium') || 
        voiceNameLower.includes('enhanced') || 
        voiceNameLower.includes('neural') ||
        voiceNameLower.includes('wavenet') ||
        voiceNameLower.includes('natural')) {
        score += 20;
    }
    
    // Prefer non-default voices (which often have better quality)
    if (!voice.default) {
        score += 10;
    }
    
    // Penalize voices that seem explicitly male
    if (voiceNameLower.includes('male') || 
        voiceNameLower.includes('man') || 
        voiceNameLower.includes('guy') ||
        voiceNameLower.includes('boy') || 
        voiceNameLower.includes('men')) {
        score -= 40; // Strong penalty for male voices
    }
    
    // Specific high-quality female voices get maximum points
    const highQualityVoices = {
        'en-US': [
            'Google US English Female',
            'Microsoft Zira Desktop',
            'Samantha',
            'Google UK English Female',
            'Microsoft Susan Mobile'
        ],
        'hi-IN': [
            'Google हिन्दी',  // Google Hindi
            'Microsoft Kalpana',
            'Microsoft Swara'
        ]
    };
    
    const langHighQualityVoices = highQualityVoices[languageCode] || [];
    if (langHighQualityVoices.includes(voice.name)) {
        score += 100; // Maximum priority for known high-quality female voices
    }
    
    return score;
};

// Function to get the preferred voice - Enhanced to prioritize high-quality female voices for the current language
const getPreferredVoice = (languageCode) => {
    const voices = window.speechSynthesis.getVoices();
    
    // If no voices available, retry
    if (voices.length === 0) {
        console.log('No voices available, trying to reload...');
        setTimeout(() => window.speechSynthesis.getVoices(), 100);
        return null;
    }
    
    console.log(`Finding female voice for language: ${languageCode}`);
    
    // Filter voices by language code (e.g., "en" or "hi")
    const langPrefix = languageCode.substring(0, 2);
    
    // Top-tier female voices by platform and language
    const preferredVoices = {
        'en': [
            'Google US English Female',   // Chrome OS/Android female voice (high quality)
            'Microsoft Zira Desktop',     // Windows female voice
            'Samantha',                   // macOS/iOS female voice
            'Google UK English Female',   // Another high-quality Google voice
            'Microsoft Hazel Desktop',    // UK English female voice
            'Microsoft Susan Mobile',     // Mobile female voice
            'Karen',                      // Australian female voice
            'Moira',                      // Irish female voice
            'Tessa',                      // South African female voice
            'Fiona',                      // Scottish female voice
            'Veena',                      // Indian female voice
            'Victoria'                    // Another female voice
        ],
        'hi': [
            'Google हिन्दी',              // Google Hindi
            'Microsoft Kalpana',          // Microsoft Hindi female voice
            'Lekha',                      // Hindi voice
            'Swara',                      // Hindi female voice
            'Heera'                       // Hindi female voice
        ]
    };
    
    // First try to find one of our preferred voices for the current language
    const currentLangPreferredVoices = preferredVoices[langPrefix] || [];
    for (const preferredVoice of currentLangPreferredVoices) {
        const voice = voices.find(v => v.name === preferredVoice);
        if (voice) {
            console.log(`Found premium female voice: ${voice.name} (${voice.lang})`);
            return voice;
        }
    }
    
    // Score all available voices and pick the best one for the current language
    let bestVoice = null;
    let bestScore = -1;
    
    for (const voice of voices) {
        const score = evaluateVoiceQuality(voice, languageCode);
        if (score > bestScore) {
            bestScore = score;
            bestVoice = voice;
        }
    }
    
    if (bestVoice) {
        console.log(`Selected best female voice: ${bestVoice.name} (${bestVoice.lang}) with score ${bestScore}`);
        return bestVoice;
    }
    
    // Last resort: any voice matching the language code
    const fallbackVoice = voices.find(v => v.lang.includes(langPrefix));
    if (fallbackVoice) {
        console.log(`Using fallback voice: ${fallbackVoice.name} (${fallbackVoice.lang}) - will adjust pitch for female tone`);
        return fallbackVoice;
    }
    
    console.warn(`No ${languageCode} voice found, using default voice with female pitch adjustment`);
    // Absolute fallback to the first available voice
    return voices[0];
};

// Test all available voices and store the best female voice for each language
const initializeVoices = () => {
    if ('speechSynthesis' in window) {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length === 0) {
            setTimeout(initializeVoices, 100);
            return;
        }
        
        // Find and store the best female voice for each language
        Object.keys(LANGUAGES).forEach(langKey => {
            const lang = LANGUAGES[langKey];
            const bestVoice = getPreferredVoice(lang.code);
            if (bestVoice) {
                lang.voiceName = bestVoice.name;
                console.log(`Set ${langKey} female voice to: ${bestVoice.name}`);
            }
        });
    }
};

// Split text into natural sentences and phrases for more realistic speech
const splitTextIntoChunks = (text) => {
    // First split by sentence endings
    const sentenceSplits = text.split(/(?<=[.!?])\s+/);
    
    const chunks = [];
    
    // For each sentence, check if it's too long and needs further splitting
    sentenceSplits.forEach(sentence => {
        if (sentence.length > 100) {
            // Split long sentences at commas, semicolons, or colons
            const phraseSplits = sentence.split(/(?<=[,;:])\s+/);
            chunks.push(...phraseSplits);
        } else {
            chunks.push(sentence);
        }
    });
    
    return chunks;
};

// Add SSML-like markup to improve speech (not actual SSML but special character based)
const addSpeechMarkup = (text) => {
    // Add slight pause after commas, longer pauses after periods
    let enhancedText = text;
    
    // Replace em dashes and double hyphens with a pause
    enhancedText = enhancedText.replace(/—|--/g, ', ');
    
    // Add emphasis to important words (this is heuristic)
    enhancedText = enhancedText.replace(
        /\b(important|critical|urgent|necessary|essential|crucial|vital|key|significant)\b/gi,
        word => word  // In a real SSML implementation, this would add emphasis tags
    );
    
    return enhancedText;
};

// Function to handle speech synthesis with enhanced realism
const speakText = (text, languageConfig, onSpeechStart, onSpeechEnd) => {
    if (!('speechSynthesis' in window)) return;

    // Cancel any ongoing speech and clear queue
    speechSynthesis.cancel();
    speechQueue = [];
    isSpeaking = false;

    // Notify that speech has started
    if (onSpeechStart) onSpeechStart();

    // Enhance text with better markers for speech
    const enhancedText = addSpeechMarkup(text);
    
    // Split into natural chunks for more realistic speaking
    const chunks = splitTextIntoChunks(enhancedText);
    
    // Get the best available voice for the current language
    const voice = getPreferredVoice(languageConfig.code);
    if (!voice) {
        console.error(`No voice available for ${languageConfig.name} speech synthesis`);
        if (onSpeechEnd) onSpeechEnd();
        return;
    }
    
    console.log(`Using female voice: ${voice.name} (${voice.lang}) for ${languageConfig.name}`);
    
    // Check if we need extra feminization based on the voice name
    const voiceNameLower = voice.name.toLowerCase();
    let extraFeminization = false;
    
    // If this doesn't appear to be a female-specific voice, bump up the pitch even more
    if (!FEMALE_IDENTIFIERS.some(term => voiceNameLower.includes(term)) && 
        !FEMALE_NAMES.some(name => voiceNameLower.includes(name))) {
        extraFeminization = true;
    }
    
    // Create utterances for each chunk with appropriate pauses
    chunks.forEach((chunk, index) => {
        const utterance = new SpeechSynthesisUtterance(chunk);
        utterance.lang = languageConfig.code;
        utterance.rate = languageConfig.rate;
        
        // Use higher pitch for non-female voices to make them sound more feminine
        utterance.pitch = extraFeminization ? 1.3 : languageConfig.pitch;
        utterance.volume = VOICE_VOLUME;
        utterance.voice = voice;
        
        // Add pauses based on punctuation - slightly shorter for faster speech
        const lastChar = chunk.trim().slice(-1);
        if (PUNCTUATION_PAUSES[lastChar]) {
            // Reduce pause duration for faster overall speech
            utterance.pauseAfter = Math.round(PUNCTUATION_PAUSES[lastChar] * 0.8);
        }
    
    // Set up event handlers
    utterance.onend = () => {
            // If there's a designated pause after this utterance
            if (utterance.pauseAfter && utterance.pauseAfter > 0) {
                setTimeout(() => {
                    isSpeaking = false;
                    speechQueue.shift();
                    
                    // If this was the last chunk, notify that speech has ended
                    if (speechQueue.length === 0 && onSpeechEnd) {
                        onSpeechEnd();
                    } else {
                        processSpeechQueue();
                    }
                }, utterance.pauseAfter);
            } else {
        isSpeaking = false;
                speechQueue.shift();
                
                // If this was the last chunk, notify that speech has ended
                if (speechQueue.length === 0 && onSpeechEnd) {
                    onSpeechEnd();
                } else {
                    processSpeechQueue();
                }
            }
    };

    utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        isSpeaking = false;
        speechQueue.shift();
            
            // Notify that speech has ended with an error
            if (speechQueue.length === 0 && onSpeechEnd) {
                onSpeechEnd();
            } else {
        processSpeechQueue();
            }
    };

    // Handle Chrome bug where speech stops after ~15 seconds
    utterance.onpause = () => {
        if (speechSynthesis.speaking) {
            speechSynthesis.resume();
        }
    };

        // Add to queue
        speechQueue.push(utterance);
    });
    
    // Start processing the queue
    processSpeechQueue();
};

// Keep speech synthesis active and prevent Chrome from cutting it off
setInterval(() => {
    // Avoid interrupted speech in Chrome
    if (speechSynthesis.speaking && speechSynthesis.paused) {
        speechSynthesis.resume();
    }
}, 100);

// Load voices when available
if ('speechSynthesis' in window) {
    // Chrome and some browsers need this event to populate the voices
    window.speechSynthesis.onvoiceschanged = () => {
        const voices = window.speechSynthesis.getVoices();
        console.log(`Loaded ${voices.length} voices for speech synthesis`);
        initializeVoices();
    };
    
    // Force load voices
    window.speechSynthesis.getVoices();
    initializeVoices();
}

// Memoized component for text inputs that preserves focus
const FocusPreservingInput = memo(({ value, onChange, placeholder, ...props }) => {
    const inputRef = useRef(null);
    
    return (
        <Input
            ref={inputRef}
            value={value}
            onChange={(e) => {
                // Only update if value actually changed to prevent unnecessary re-renders
                if (e.target.value !== value) {
                    onChange(e);
                }
            }}
            placeholder={placeholder}
            onFocus={() => {
                // Store the cursor position
                const pos = inputRef.current?.selectionStart;
                // Restore it after the state update
                setTimeout(() => {
                    if (inputRef.current) {
                        inputRef.current.selectionStart = pos;
                        inputRef.current.selectionEnd = pos;
                    }
                }, 0);
            }}
            {...props}
        />
    );
});

// Memoized component for textareas that preserves focus
const FocusPreservingTextarea = memo(({ value, onChange, placeholder, ...props }) => {
    const textareaRef = useRef(null);
    
    return (
        <Textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => {
                // Only update if value actually changed
                if (e.target.value !== value) {
                    onChange(e);
                }
            }}
            placeholder={placeholder}
            onFocus={() => {
                // Store the cursor position
                const pos = textareaRef.current?.selectionStart;
                // Restore it after the state update
                setTimeout(() => {
                    if (textareaRef.current) {
                        textareaRef.current.selectionStart = pos;
                        textareaRef.current.selectionEnd = pos;
                    }
                }, 0);
            }}
            {...props}
        />
    );
});

// Pop-in animation keyframes
const popIn = keyframes`
  0% { transform: scale(0.7); opacity: 0; }
  80% { transform: scale(1.05); opacity: 1; }
  100% { transform: scale(1); opacity: 1; }
`;

function Message({ text, sender, languageConfig, onSpeakingStateChange }) {
    // Add speech synthesis to each message with speech state handling
    const speakMessage = useCallback(() => {
        if (sender === 'bot') {
            speakText(
                text, 
                languageConfig,
                () => onSpeakingStateChange && onSpeakingStateChange(true),
                () => onSpeakingStateChange && onSpeakingStateChange(false)
            );
        }
    }, [text, sender, languageConfig, onSpeakingStateChange]);

    return (
        <Flex
            w="100%"
            justify={sender === 'user' ? 'flex-end' : 'flex-start'}
            mb={4}
        >
            <HStack
                spacing={2}
                maxW="80%"
                alignItems="flex-start"
                flexDirection={sender === 'user' ? 'row-reverse' : 'row'}
            >
                <Avatar 
                    size="md"
                    src={sender === 'user' ? USER_AVATAR : BOT_AVATAR}
                    name={sender === 'user' ? 'User' : 'NeuroSri'}
                    bg={sender === 'user' ? 'linear-gradient(135deg, #4169E1 0%, #9370DB 50%, #8A2BE2 100%)' : 'linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #4169E1 100%)'}
                    showBorder={true}
                    borderColor={sender === 'user' ? 'yellow.200' : 'purple.200'}
                    borderWidth="2px"
                    ignoreFallback={true}
                />
                <Box
                    onClick={sender === 'bot' ? speakMessage : undefined}
                    cursor={sender === 'bot' ? 'pointer' : 'default'}
                    bg={sender === 'user' 
                        ? 'linear-gradient(135deg, #4169E1 0%, #9370DB 50%, #8A2BE2 100%)' 
                        : 'linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #4169E1 100%)'
                    }
                    color="white"
                    px={5}
                    py={3.5}
                    borderRadius="2xl"
                    maxW="100%"
                    position="relative"
                    fontFamily="body"
                    fontSize="md"
                    fontWeight="medium"
                    textShadow="0 1px 2px rgba(0,0,0,0.08)"
                    boxShadow="0 4px 16px 0 rgba(0,0,0,0.10), 0 1.5px 4px 0 rgba(0,0,0,0.08)"
                    animation={`${popIn} 0.35s cubic-bezier(0.23, 1.15, 0.4, 1) both`}
                    _hover={sender === 'bot' ? {
                        transform: 'scale(1.03)',
                        transition: 'transform 0.2s ease-in-out',
                        boxShadow: '0 6px 18px rgba(0,0,0,0.18)'
                    } : undefined}
                    title={sender === 'bot' ? 'Click to hear this message' : ''}
                    _before={{
                        content: '""',
                        position: 'absolute',
                        top: '18px',
                        [sender === 'user' ? 'right' : 'left']: '-10px',
                        borderWidth: '10px',
                        borderStyle: 'solid',
                        borderColor: 'transparent',
                        [sender === 'user' ? 'borderLeftColor' : 'borderRightColor']: sender === 'user' ? '#4169E1' : '#8A2BE2',
                        opacity: 0.5
                    }}
                >
                    <Text>{text}</Text>
                </Box>
            </HStack>
        </Flex>
    );
}

// User Info Form Component - moved outside ChatWindow
const UserInfoForm = ({ 
    userInfo, 
    handleUserInfoChange, 
    handleUserFormSubmit, 
    nameInputRef, 
    ageInputRef, 
    mainConcernRef, 
    medicationsRef 
}) => (
    <Box 
        as="form" 
        onSubmit={handleUserFormSubmit} 
        width="100%" 
        p={4} 
        borderRadius="lg" 
        bg="white" 
        boxShadow="md"
    >
        <VStack spacing={4} align="stretch">
            <Heading size="md" color="blue.600" textAlign="center">
                Welcome to NeuroSri
            </Heading>
            <Text textAlign="center" fontSize="sm" color="gray.600">
                Please provide some information to help us personalize your experience
            </Text>
            
            <Divider />
            
            <FormControl isRequired>
                <FormLabel>Your Name</FormLabel>
                <FocusPreservingInput 
                    ref={nameInputRef}
                    placeholder="Enter your name" 
                    value={userInfo.name} 
                    onChange={handleUserInfoChange('name')} 
                />
            </FormControl>
            
            <HStack spacing={8}>
                <FormControl isRequired>
                    <FormLabel>Age</FormLabel>
                    <NumberInput 
                        min={5} 
                        max={100} 
                        value={userInfo.age} 
                        onChange={handleUserInfoChange('age')}
                    >
                        <NumberInputField 
                            ref={ageInputRef} 
                            placeholder="Age" 
                        />
                        <NumberInputStepper>
                            <NumberIncrementStepper />
                            <NumberDecrementStepper />
                        </NumberInputStepper>
                    </NumberInput>
                </FormControl>
                
                <FormControl>
                    <FormLabel>Gender</FormLabel>
                    <RadioGroup value={userInfo.gender} onChange={handleUserInfoChange('gender')}>
                        <Stack direction="row">
                            <Radio value="male">Male</Radio>
                            <Radio value="female">Female</Radio>
                            <Radio value="other">Other</Radio>
                        </Stack>
                    </RadioGroup>
                </FormControl>
            </HStack>
            
            <FormControl isRequired>
                <FormLabel>What brings you here today?</FormLabel>
                <FocusPreservingTextarea 
                    ref={mainConcernRef}
                    placeholder="Please describe your main concerns or what you'd like help with" 
                    value={userInfo.mainConcern} 
                    onChange={handleUserInfoChange('mainConcern')}
                    rows={3}
                />
            </FormControl>
            
            <HStack spacing={6}>
                <FormControl>
                    <FormLabel>Previous therapy experience?</FormLabel>
                    <Select value={userInfo.previousTherapy} onChange={handleUserInfoChange('previousTherapy')}>
                        <option value="no">No previous experience</option>
                        <option value="some">Some experience</option>
                        <option value="extensive">Extensive experience</option>
                    </Select>
                </FormControl>
                
                <FormControl>
                    <FormLabel>Sleep quality recently</FormLabel>
                    <Select value={userInfo.sleepQuality} onChange={handleUserInfoChange('sleepQuality')}>
                        <option value="poor">Poor</option>
                        <option value="fair">Fair</option>
                        <option value="good">Good</option>
                        <option value="excellent">Excellent</option>
                    </Select>
                </FormControl>
            </HStack>
            
            <FormControl>
                <FormLabel>Previous neurological history</FormLabel>
                <Select value={userInfo.neurologicalHistory} onChange={handleUserInfoChange('neurologicalHistory')}>
                    <option value="none">None</option>
                    <option value="headaches">Frequent headaches</option>
                    <option value="migraines">Migraines</option>
                    <option value="seizures">Seizures</option>
                    <option value="concussion">Previous concussion</option>
                    <option value="tbi">Traumatic brain injury</option>
                    <option value="stroke">Stroke</option>
                    <option value="other">Other (specify in concerns)</option>
                </Select>
            </FormControl>
            
            <FormControl>
                <FormLabel>Current medications (optional)</FormLabel>
                <FocusPreservingTextarea 
                    ref={medicationsRef}
                    placeholder="List any current medications you are taking" 
                    value={userInfo.medications} 
                    onChange={handleUserInfoChange('medications')}
                    rows={2}
                />
            </FormControl>
            
            <Button 
                type="submit" 
                colorScheme="blue" 
                rightIcon={<FaPaperPlane />}
                size="lg"
                mt={2}
            >
                Start Session
            </Button>
            
            <Text fontSize="xs" color="gray.500" textAlign="center">
                Your information helps us provide personalized support but will not be shared externally.
            </Text>
        </VStack>
    </Box>
);

function ChatWindow({ currentEmotion, emotionData }) {
    const { colorMode } = useColorMode();
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [showUserForm, setShowUserForm] = useState(true);
    const [hasShownSetup, setHasShownSetup] = useState(false);
    const [hasShownInitial, setHasShownInitial] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [currentLanguage, setCurrentLanguage] = useState(LANGUAGES.ENGLISH);
    const [isSpeaking, setIsSpeakingState] = useState(false);
    
    // Create refs for input fields to maintain focus
    const nameInputRef = useRef(null);
    const ageInputRef = useRef(null);
    const mainConcernRef = useRef(null);
    const medicationsRef = useRef(null);
    
    const [userInfo, setUserInfo] = useState({
        name: '',
        age: '',
        gender: 'male',
        mainConcern: '',
        previousTherapy: 'no',
        sleepQuality: 'good',
        neurologicalHistory: 'none',
        medications: ''
    });
    const toast = useToast();

    // Function to handle user form input changes
    const handleUserInfoChange = (field) => (eventOrValue) => {
        const value = eventOrValue.target ? eventOrValue.target.value : eventOrValue;
        // Only update state if the value has actually changed
        setUserInfo((prev) => {
            if (prev[field] === value) return prev; // No change, return previous state
            return { ...prev, [field]: value }; // Update with new value
        });
    };

    // Function to handle user form submission
    const handleUserFormSubmit = async (e) => {
        e.preventDefault();
        
        // Validate form
        if (!userInfo.name.trim() || !userInfo.age || !userInfo.mainConcern.trim()) {
            toast({
                title: "Missing Information",
                description: "Please fill out all required fields.",
                status: "warning",
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        try {
            // Send user info to backend
            const response = await api.submitUserInfo(userInfo);
            
            if (response && response.success) {
                // Hide the form and show the chat interface
                setShowUserForm(false);
                
                // Add a welcome message that uses the user's name
                const welcomeMessage = { 
                    text: `Hello ${userInfo.name}! I'm NeuroSri, your mental health AI counselor. I'll be analyzing your EEG signals to better understand your emotional state. Thank you for sharing your information about ${userInfo.mainConcern}. How are you feeling right now?`, 
                    sender: 'bot' 
                };
                
                setMessages([welcomeMessage]);
                
                // Speak the welcome message
                speakText(
                    welcomeMessage.text, 
                    currentLanguage, 
                    () => setIsSpeakingState(true), 
                    () => setIsSpeakingState(false)
                );
                
                toast({
                    title: "Information Saved",
                    description: "Thank you for sharing your information.",
                    status: "success",
                    duration: 3000,
                    isClosable: true,
                });
            } else if (response && response.error) {
                // Show error but continue anyway with local data
                toast({
                    title: "Warning",
                    description: `${response.error} Continuing with local data only.`,
                    status: "warning",
                    duration: 5000,
                    isClosable: true,
                });
                
                // Continue with local data only
                setShowUserForm(false);
                
                const welcomeMessage = { 
                    text: `Hello ${userInfo.name}! I'm NeuroSri, your mental health AI counselor. Thank you for sharing your information about ${userInfo.mainConcern}. How are you feeling right now?`, 
                    sender: 'bot' 
                };
                
                setMessages([welcomeMessage]);
                speakText(
                    welcomeMessage.text, 
                    currentLanguage, 
                    () => setIsSpeakingState(true), 
                    () => setIsSpeakingState(false)
                );
            } else {
                throw new Error("Unknown error occurred");
            }
        } catch (error) {
            console.error('Error submitting user info:', error);
            
            // Show error but continue anyway with local data
            toast({
                title: "Connection Error",
                description: "Could not connect to the server. Continuing with local data only.",
                status: "error",
                duration: 5000,
                isClosable: true,
            });
            
            // Continue with local data only since server is unreachable
            setShowUserForm(false);
            
            const welcomeMessage = { 
                text: `Hello ${userInfo.name}! I'm NeuroSri, your mental health AI counselor working in offline mode. Thank you for sharing your information. How are you feeling right now?`, 
                sender: 'bot' 
            };
            
            setMessages([welcomeMessage]);
            speakText(
                welcomeMessage.text, 
                currentLanguage, 
                () => setIsSpeakingState(true), 
                () => setIsSpeakingState(false)
            );
        }
    };

    // Function to stop chatbot speech
    const stopSpeech = () => {
        if (speechSynthesis.speaking || speechQueue.length > 0 || isSpeaking) {
            console.log('Stopping chatbot speech immediately');
            
            // Use our global function to ensure speech is stopped
            ensureSpeechStopped();
            
            // Update React state
            setIsSpeakingState(false);
            
            toast({
                title: "Voice Stopped",
                description: "Ready for your voice input.",
                status: "info",
                duration: 1000,
                isClosable: true,
            });
        }
    };

    // Initialize speech recognition
    const [recognition, setRecognition] = useState(null);

    // Setup speech recognition with current language
    useEffect(() => {
        if ('webkitSpeechRecognition' in window) {
            // If there's an existing recognition instance, stop it
            if (recognition) {
                recognition.stop();
                setIsListening(false);
            }
            
            const newRecognition = new window.webkitSpeechRecognition();
            newRecognition.continuous = false;
            newRecognition.interimResults = false;
            newRecognition.lang = currentLanguage.code;

            console.log(`Speech recognition set to ${currentLanguage.name} (${currentLanguage.code})`);

            newRecognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                setInput(transcript);
                // Automatically submit after voice input
                handleSubmit(new Event('submit'), transcript);
            };

            newRecognition.onerror = (event) => {
                console.error(`Speech recognition error: ${event.error} (${currentLanguage.name})`);
                setIsListening(false);
                toast({
                    title: "Voice Input Error",
                    description: `There was an error with ${currentLanguage.name} voice input. Please try again or use text input.`,
                    status: "error",
                    duration: 3000,
                    isClosable: true,
                });
            };

            newRecognition.onend = () => {
                setIsListening(false);
            };

            setRecognition(newRecognition);
        }
    }, [currentLanguage]);

    // Handle voice input
    const toggleVoiceInput = () => {
        // First stop any ongoing speech from the chatbot
        if (speechSynthesis.speaking || speechQueue.length > 0 || isSpeaking) {
            console.log('Stopping chatbot speech before voice input');
            
            // Use our global function to ensure speech is stopped
            ensureSpeechStopped();
            
            // Update React state
            setIsSpeakingState(false);
        }

        if (!recognition) {
            toast({
                title: "Voice Input Not Available",
                description: "Your browser doesn't support voice input. Please use text input instead.",
                status: "warning",
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        if (isListening) {
            recognition.stop();
        } else {
            // Start voice recognition immediately - remove previous delay
            try {
            recognition.start();
            setIsListening(true);
                
                toast({
                    title: `${currentLanguage.name} Voice Input Active`,
                    description: `Listening for ${currentLanguage.name} speech...`,
                    status: "info",
                    duration: 1500,
                    isClosable: true,
                });
            } catch (error) {
                console.error('Error starting voice recognition:', error);
                toast({
                    title: "Voice Input Error",
                    description: "Couldn't start voice recognition. Please try again.",
                    status: "error",
                    duration: 2000,
                    isClosable: true,
                });
            }
        }
    };

    // Handle language change
    const handleLanguageChange = (e) => {
        const selectedLang = e.target.value;
        const newLanguage = selectedLang === 'hindi' ? LANGUAGES.HINDI : LANGUAGES.ENGLISH;
        
        // Stop any ongoing speech before changing language
        if (speechSynthesis.speaking || speechQueue.length > 0 || isSpeaking) {
            console.log('Stopping chatbot speech before language change');
            ensureSpeechStopped();
            setIsSpeakingState(false);
        }
        
        setCurrentLanguage(newLanguage);
        
        toast({
            title: `Language Changed to ${newLanguage.name}`,
            description: `Voice input and output will now use ${newLanguage.name} with female voice.`,
            status: "success",
            duration: 2000,
            isClosable: true,
        });
    };

    // Speak the bot's response
    const speakResponse = useCallback((text) => {
        speakText(
            text, 
            currentLanguage, 
            () => setIsSpeakingState(true),   // onSpeechStart
            () => setIsSpeakingState(false)   // onSpeechEnd
        );
    }, [currentLanguage]);

    // Handle message submission
    const handleSubmit = async (e, voiceInput = null) => {
        e.preventDefault();
        const messageText = voiceInput || input;
        if (!messageText.trim()) return;

        const userMessage = { text: messageText, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setInput('');

        try {
            // Include user info with message
            const response = await api.sendMessage(messageText, userInfo);
            
            if (response && response.response) {
                const botMessage = { text: response.response, sender: 'bot' };
                setMessages(prev => [...prev, botMessage]);
                // Automatically speak bot's response
                speakResponse(response.response);
            } else if (response && response.error) {
                toast({
                    title: "Error",
                    description: response.error,
                    status: "error",
                    duration: 3000,
                    isClosable: true,
                });
                
                // Add fallback response so conversation can continue
                const fallbackMessage = { 
                    text: "I'm sorry, I couldn't process your message due to a connection issue. Could you try again?", 
                    sender: 'bot' 
                };
                setMessages(prev => [...prev, fallbackMessage]);
                speakResponse(fallbackMessage.text);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            
            // Add fallback response so conversation can continue
            const fallbackMessage = { 
                text: "I'm sorry, I couldn't connect to my server. I'm working in offline mode right now. Could you try again later?", 
                sender: 'bot' 
            };
            setMessages(prev => [...prev, fallbackMessage]);
            speakResponse(fallbackMessage.text);
            
            toast({
                title: "Connection Error",
                description: "Failed to connect to the server. Working in offline mode.",
                status: "error",
                duration: 3000,
                isClosable: true,
            });
        }
    };

    // Skip the initial setup and greeting messages
    useEffect(() => {
            setHasShownSetup(true);
        setHasShownInitial(true);
    }, []);

    // Ensure voices are loaded when component mounts and initialize female voices
    useEffect(() => {
        // Load and initialize voices
        if ('speechSynthesis' in window) {
            // Force load voices
            const loadVoices = () => {
                const voices = window.speechSynthesis.getVoices();
                if (voices.length === 0) {
                    // If no voices are available yet, try again in 100ms
                    setTimeout(loadVoices, 100);
                } else {
                    console.log(`Loaded ${voices.length} voices`);
                    
                    // Log available voices for debugging
                    console.log("Available voices for female selection:");
                    voices.forEach(voice => {
                        // Check if this is likely a female voice
                        const voiceNameLower = voice.name.toLowerCase();
                        const isFemaleVoice = 
                            FEMALE_IDENTIFIERS.some(term => voiceNameLower.includes(term)) ||
                            FEMALE_NAMES.some(name => voiceNameLower.includes(name));
                            
                        console.log(`Voice: ${voice.name}, Lang: ${voice.lang}, Female: ${isFemaleVoice ? 'Yes' : 'No'}`);
                    });
                    
                    // Check Hindi voice availability
                    const hindiVoices = voices.filter(v => v.lang.includes('hi'));
                    if (hindiVoices.length > 0) {
                        console.log(`Found ${hindiVoices.length} Hindi voice(s):`);
                        hindiVoices.forEach(v => console.log(`- ${v.name} (${v.lang})`));
                    } else {
                        console.warn("No Hindi voices found, will use pitch adjustment for female voice");
                    }
                    
                    // Initialize the best female voices for each language
                    initializeVoices();
                    
                    // Test the voice settings with a silent utterance
                    const testVoice = new SpeechSynthesisUtterance("");
                    testVoice.volume = 0;
                    window.speechSynthesis.speak(testVoice);
                }
            };
            
            loadVoices();
        }
    }, []);

    // Create the user context prompt with the updated fields
    const _create_user_context_prompt = () => {
        const profile = userInfo;
        
        // Format information about the user
        let user_context = (
            `User Profile Information:\n`+
            `- Name: ${profile.name}\n`+
            `- Age: ${profile.age}\n`+
            `- Gender: ${profile.gender}\n`+
            `- Main concern: ${profile.mainConcern}\n`
        );
        
        // Add additional details if available
        if (profile.previousTherapy) {
            user_context += `- Previous therapy experience: ${profile.previousTherapy}\n`;
        }
        
        if (profile.sleepQuality) {
            user_context += `- Sleep quality: ${profile.sleepQuality}\n`;
        }
        
        if (profile.neurologicalHistory && profile.neurologicalHistory !== 'none') {
            user_context += `- Neurological history: ${profile.neurologicalHistory}\n`;
        }
        
        if (profile.medications && profile.medications.trim()) {
            user_context += `- Current medications: ${profile.medications}\n`;
        }
        
        return user_context;
    };

    return (
        <Box 
            borderWidth={1} 
            borderRadius="lg" 
            p={4} 
            bgGradient={colorMode === 'dark' 
                ? 'linear(120deg, gray.700 0%, white 50%, white 30%, gray.700 100%)' 
                : 'linear(120deg, cyan.100 0%, white 50%, white 30%, cyan.100 100%)'}
            height="600px"
            boxShadow="xl"
        >
            <VStack h="100%" spacing={4}>
                {!showUserForm && (
                    <HStack w="100%" justifyContent="space-between">
                        <Select 
                            size="sm" 
                            width="150px" 
                            value={currentLanguage === LANGUAGES.HINDI ? 'hindi' : 'english'}
                            onChange={handleLanguageChange}
                            icon={<FaLanguage />}
                            bg="white"
                        >
                            <option value="english">English</option>
                            <option value="hindi">Hindi</option>
                        </Select>
                        <Text fontSize="xs" color="white">
                            Female Voice: {currentLanguage.name}
                        </Text>
                    </HStack>
                )}
                
                {showUserForm ? (
                    <UserInfoForm 
                        userInfo={userInfo}
                        handleUserInfoChange={handleUserInfoChange}
                        handleUserFormSubmit={handleUserFormSubmit}
                        nameInputRef={nameInputRef}
                        ageInputRef={ageInputRef}
                        mainConcernRef={mainConcernRef}
                        medicationsRef={medicationsRef}
                    />
                ) : (
                    <>
                    <Box 
                        flex="1" 
                        w="100%" 
                        overflowY="auto" 
                        css={{
                            '&::-webkit-scrollbar': {
                                width: '4px',
                            },
                            '&::-webkit-scrollbar-track': {
                                width: '6px',
                                background: 'rgba(255, 255, 255, 0.1)',
                            },
                            '&::-webkit-scrollbar-thumb': {
                                background: 'rgba(255, 255, 255, 0.3)',
                                borderRadius: '24px',
                            },
                        }}
                        p={2}
                    >
                        {messages.map((message, index) => (
                            <Message 
                                key={index}
                                text={message.text}
                                sender={message.sender}
                                languageConfig={currentLanguage}
                                onSpeakingStateChange={setIsSpeakingState}
                            />
                        ))}
                    </Box>

                    <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                        <HStack>
                            <Input
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder={`Type your message in ${currentLanguage.name}...`}
                                size="md"
                                bg="white"
                                color="gray.800"
                            />
                            
                            {isSpeaking && (
                                <Tooltip label="Stop chatbot speaking">
                                    <IconButton
                                        aria-label="Stop speech"
                                        icon={<FaVolumeMute />}
                                        onClick={stopSpeech}
                                        colorScheme="orange"
                                    />
                                </Tooltip>
                            )}
                            
                            <Tooltip label={isListening ? 
                                `Stop ${currentLanguage.name} voice input` : 
                                `Start ${currentLanguage.name} voice input`}
                            >
                                <IconButton
                                    aria-label={`${currentLanguage.name} voice input`}
                                    icon={isListening ? <FaStop /> : <FaMicrophone />}
                                    onClick={toggleVoiceInput}
                                    colorScheme={isListening ? "red" : "blue"}
                                />
                            </Tooltip>
                            <Button colorScheme="blue" type="submit">
                                Send
                            </Button>
                        </HStack>
                    </form>
                    </>
                )}
            </VStack>
        </Box>
    );
}

export default ChatWindow;