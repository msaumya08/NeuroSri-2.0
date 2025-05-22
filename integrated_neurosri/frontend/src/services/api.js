import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';  // Update this to match your backend URL

// Add axios interceptor for debugging
axios.interceptors.request.use(request => {
    console.log('Starting Request:', request);
    return request;
});

axios.interceptors.response.use(
    response => {
        console.log('Response:', response);
        return response;
    },
    error => {
        console.error('API Error:', error.message);
        if (error.code === 'ERR_NETWORK') {
            console.error('Network Error: Unable to connect to the backend server. Please ensure the server is running at', API_BASE_URL);
        }
        return Promise.reject(error);
    }
);

export const api = {
    // Get emotion data from backend
    getEmotion: async () => {
        try {
            console.log('Fetching emotion data from:', `${API_BASE_URL}/api/emotion`);
            const response = await axios.get(`${API_BASE_URL}/api/emotion`);
            console.log('Emotion data received:', response.data);
            return response.data;
        } catch (error) {
            console.error('Error fetching emotion:', error);
            // Return mock data for development
            return {
                emotion: 'neutral',
                confidence: 0.5,
                eeg_data: [],
                is_setup_phase: false,
                setup_complete: true
            };
        }
    },

    // Send message to chatbot
    sendMessage: async (message, userInfo) => {
        try {
            console.log('Sending message to:', `${API_BASE_URL}/api/chat`);
            const response = await axios.post(`${API_BASE_URL}/api/chat`, {
                message,
                user_info: userInfo
            });
            
            console.log('Chat response received:', response.data);
            
            if (response.data && response.data.response) {
                return {
                    response: response.data.response,
                    emotion: response.data.emotion || 'neutral',
                    confidence: response.data.confidence || 0.5
                };
            }
            return response.data;
        } catch (error) {
            console.error('Error sending message:', error);
            return {
                error: 'Failed to connect to the server. Please ensure the backend server is running at ' + API_BASE_URL
            };
        }
    },

    // Submit user information
    submitUserInfo: async (userInfo) => {
        try {
            console.log('Submitting user info to:', `${API_BASE_URL}/api/user-info`);
            const response = await axios.post(`${API_BASE_URL}/api/user-info`, userInfo);
            console.log('User info response:', response.data);
            return response.data;
        } catch (error) {
            console.error('Error submitting user info:', error);
            return {
                error: 'Failed to save user information. Please ensure the backend server is running at ' + API_BASE_URL
            };
        }
    },

    getEEGData: async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/api/eeg-data`);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            return { error: 'Failed to fetch EEG data.' };
        }
    }
}; 