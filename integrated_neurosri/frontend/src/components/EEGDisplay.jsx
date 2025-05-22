import React, { useState, useEffect, useCallback } from 'react';
import { Box, Text, Center, Spinner, Heading, useColorMode } from '@chakra-ui/react';
import Plot from 'react-plotly.js';

const BUFFER_SIZE = 200; // Number of points to show in the plot

// Function to generate random number within a range
const random = (min, max) => Math.random() * (max - min) + min;

// Function to add random walk to simulate baseline drift
let baselineDrift = 0;
const updateBaseline = () => {
    baselineDrift += random(-1, 1);
    baselineDrift = Math.max(Math.min(baselineDrift, 50), -50); // Limit drift range
    return baselineDrift;
};

// Function to generate realistic-looking brain wave patterns
const generateBrainWave = (t, channel) => {
    // Random frequency jitter
    const freqJitter = random(0.9, 1.1);
    
    // Random amplitude modulation
    const amplitudeModulation = (Math.sin(t / 50) + 1) / 2 * random(0.8, 1.2);
    
    // Base frequencies with slight difference between channels
    const baseFreq = channel === 1 ? 10 : channel === 2 ? 11 : 12;
    
    // Alpha waves (8-13 Hz) with random phase
    const alphaPhase = random(0, 2 * Math.PI);
    const alpha = Math.sin(2 * Math.PI * baseFreq * freqJitter * t / BUFFER_SIZE + alphaPhase) * 80 * amplitudeModulation;
    
    // Beta waves (13-30 Hz) with varying amplitude
    const betaAmp = random(30, 70);
    const beta = Math.sin(2 * Math.PI * 25 * freqJitter * t / BUFFER_SIZE) * betaAmp * amplitudeModulation;
    
    // Theta waves (4-8 Hz) with random phase
    const thetaPhase = random(0, 2 * Math.PI);
    const theta = Math.sin(2 * Math.PI * 6 * freqJitter * t / BUFFER_SIZE + thetaPhase) * 60 * amplitudeModulation;
    
    // Delta waves (1-4 Hz) for slow variations
    const delta = Math.sin(2 * Math.PI * 2 * t / BUFFER_SIZE) * 40 * random(0.9, 1.1);
    
    // Enhanced random noise with occasional spikes
    const baseNoise = (Math.random() - 0.5) * 30;
    const spike = Math.random() < 0.02 ? random(-50, 50) : 0; // Random spikes
    const noise = baseNoise + spike;
    
    // Add baseline drift
    const drift = updateBaseline();
    
    // Combine all components with varying weights
    const signal = (
        alpha * 1.0 + 
        beta * 0.6 + 
        theta * 0.4 + 
        delta * 0.3 + 
        noise * 0.7 + 
        drift
    );
    
    return signal;
};

// Function to generate a complete wave pattern
const generateWavePattern = (channel) => {
    const pattern = [];
    for (let i = 0; i < BUFFER_SIZE; i++) {
        pattern.push(generateBrainWave(i, channel));
    }
    return pattern;
};

function EEGDisplay({ data }) {
    const { colorMode } = useColorMode();
    const [plotData, setPlotData] = useState([
        {
            x: Array.from({ length: BUFFER_SIZE }, (_, i) => i),
            y: generateWavePattern(1),
            type: 'scatter',
            mode: 'lines',
            name: 'Channel 1',
            line: {
                color: 'rgba(33, 150, 243, 0.8)',
                width: 2,
                shape: 'spline',
                smoothing: 1.3
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(33, 150, 243, 0.1)',
            subplot: 'subplot1'
        },
        {
            x: Array.from({ length: BUFFER_SIZE }, (_, i) => i),
            y: generateWavePattern(2),
            type: 'scatter',
            mode: 'lines',
            name: 'Channel 2',
            line: {
                color: 'rgba(76, 175, 80, 0.8)',
                width: 2,
                shape: 'spline',
                smoothing: 1.3
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(76, 175, 80, 0.1)',
            yaxis: 'y2',
            subplot: 'subplot2'
        },
        {
            x: Array.from({ length: BUFFER_SIZE }, (_, i) => i),
            y: generateWavePattern(3),
            type: 'scatter',
            mode: 'lines',
            name: 'Channel 3',
            line: {
                color: 'rgba(255, 87, 34, 0.8)',
                width: 2,
                shape: 'spline',
                smoothing: 1.3
            },
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 87, 34, 0.1)',
            yaxis: 'y3',
            subplot: 'subplot3'
        }
    ]);

    const updatePlotData = useCallback(() => {
        setPlotData(currentPlotData => {
            const [channel1, channel2, channel3] = currentPlotData;

            // Generate new wave patterns with different channels
            const newChannel1Y = [...channel1.y.slice(1), generateBrainWave(channel1.x[BUFFER_SIZE - 1], 1)];
            const newChannel2Y = [...channel2.y.slice(1), generateBrainWave(channel2.x[BUFFER_SIZE - 1], 2)];
            const newChannel3Y = [...channel3.y.slice(1), generateBrainWave(channel3.x[BUFFER_SIZE - 1], 3)];

            // Update x values by incrementing each value by 1
            const newX = channel1.x.map(x => x + 1);

            return [
                {
                    ...channel1,
                    x: newX,
                    y: newChannel1Y,
                },
                {
                    ...channel2,
                    x: newX,
                    y: newChannel2Y,
                },
                {
                    ...channel3,
                    x: newX,
                    y: newChannel3Y,
                }
            ];
        });
    }, []);

    // Update the plot every 50ms for smooth and slower animation
    useEffect(() => {
        const interval = setInterval(updatePlotData, 50);
        return () => clearInterval(interval);
    }, [updatePlotData]);

    const layout = {
        title: {
            text: 'Real-time Brain Activity',
            font: {
                size: 24,
                color: '#2c3e50'
            }
        },
        autosize: true,
        height: 450,
        margin: { t: 60, b: 40, l: 50, r: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        showlegend: false,
        grid: {
            rows: 3,
            columns: 1,
            pattern: 'independent',
            roworder: 'top to bottom'
        },
        xaxis: {
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: false,
            range: [plotData[0].x[0], plotData[0].x[BUFFER_SIZE - 1]],
            domain: [0, 1],
            showticklabels: false
        },
        yaxis: {
            title: {
                text: 'Channel 1 (μV)',
                font: {
                    size: 14,
                    color: '#2196F3'
                }
            },
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: true,
            zerolinecolor: 'rgba(128,128,128,0.2)',
            range: [-300, 300],
            domain: [0.7, 1]
        },
        xaxis2: {
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: false,
            range: [plotData[0].x[0], plotData[0].x[BUFFER_SIZE - 1]],
            domain: [0, 1]
        },
        yaxis2: {
            title: {
                text: 'Channel 2 (μV)',
                font: {
                    size: 14,
                    color: '#4CAF50'
                }
            },
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: true,
            zerolinecolor: 'rgba(128,128,128,0.2)',
            range: [-300, 300],
            domain: [0.35, 0.65]
        },
        xaxis3: {
            title: {
                text: 'Time (samples)',
                font: {
                    size: 14,
                    color: '#2c3e50'
                }
            },
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: false,
            range: [plotData[0].x[0], plotData[0].x[BUFFER_SIZE - 1]],
            domain: [0, 1]
        },
        yaxis3: {
            title: {
                text: 'Channel 3 (μV)',
                font: {
                    size: 14,
                    color: '#FF5722'
                }
            },
            showgrid: true,
            gridcolor: 'rgba(128,128,128,0.15)',
            zeroline: true,
            zerolinecolor: 'rgba(128,128,128,0.2)',
            range: [-300, 300],
            domain: [0, 0.3]
        }
    };

    const config = {
        responsive: true,
        displayModeBar: false,
        displaylogo: false,
        scrollZoom: false,
        staticPlot: true
    };

    return (
        <Box 
            w="90%" 
            p={3} 
            mx="auto"
            borderRadius="lg" 
            bg={colorMode === 'dark' ? '#C0C0C0' : undefined}
            bgGradient={colorMode === 'dark' ? undefined : 'linear(to-b, cyan.100, white)'}
            boxShadow="sm"
            position="relative"
        >
            <Heading size="md" mb={4} textAlign="center" color="gray.700">
                Brain Activity Monitor
            </Heading>
            {!data ? (
                <Center h="300px">
                    <Spinner size="xl" color="blue.500" />
                    <Text ml={4} color="gray.500">Initializing EEG monitor...</Text>
                </Center>
            ) : (
                <Plot
                    data={plotData}
                    layout={layout}
                    config={config}
                    style={{ width: '100%', height: '100%' }}
                    useResizeHandler={true}
                />
            )}
        </Box>
    );
}

export default EEGDisplay;