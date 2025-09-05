// Acoustic Analysis System
class AcousticAnalyzer {
    constructor() {
        this.sampleRate = 48000;
        this.fftSize = 2048;
        this.frequencies = new Float32Array(this.fftSize);
        this.initializeFrequencies();
    }

    initializeFrequencies() {
        for (let i = 0; i < this.fftSize; i++) {
            this.frequencies[i] = i * this.sampleRate / this.fftSize;
        }
    }

    // FFT Berechnung
    calculateFFT(audioData) {
        // FFT-Implementation
        return new Float32Array(this.fftSize/2);
    }

    // HRTF-Berechnung
    calculateHRTF(azimuth, elevation) {
        const hrtfResponse = new Float32Array(this.fftSize);
        // HRTF-Modell-Implementation
        return hrtfResponse;
    }

    // Raumakustik-Simulation
    simulateRoom(dimensions, materials) {
        const roomResponse = {
            reflections: [],
            rt60: 0,
            modes: []
        };

        // Raummoden-Berechnung
        const modes = this.calculateRoomModes(dimensions);
        
        // Nachhallzeit-Berechnung
        const rt60 = this.calculateRT60(dimensions, materials);
        
        // Erste Reflexionen
        const reflections = this.calculateEarlyReflections(dimensions, materials);

        return {
            modes,
            rt60,
            reflections
        };
    }

    // Raummoden-Berechnung
    calculateRoomModes(dimensions) {
        const modes = [];
        const c = 343; // Schallgeschwindigkeit in m/s

        for (let nx = 0; nx <= 5; nx++) {
            for (let ny = 0; ny <= 5; ny++) {
                for (let nz = 0; nz <= 5; nz++) {
                    if (nx === 0 && ny === 0 && nz === 0) continue;

                    const freq = (c/2) * Math.sqrt(
                        Math.pow(nx/dimensions.length, 2) +
                        Math.pow(ny/dimensions.width, 2) +
                        Math.pow(nz/dimensions.height, 2)
                    );

                    modes.push({
                        frequency: freq,
                        indices: [nx, ny, nz]
                    });
                }
            }
        }

        return modes.sort((a, b) => a.frequency - b.frequency);
    }

    // Nachhallzeit-Berechnung (Sabine-Formel)
    calculateRT60(dimensions, materials) {
        const volume = dimensions.length * dimensions.width * dimensions.height;
        const surfaces = this.calculateSurfaces(dimensions);
        
        // Mittlere Absorption berechnen
        let totalAbsorption = 0;
        let totalArea = 0;
        
        for (const [surface, area] of Object.entries(surfaces)) {
            totalAbsorption += area * (materials[surface]?.absorption || 0.1);
            totalArea += area;
        }
        
        const meanAlpha = totalAbsorption / totalArea;
        
        // Sabine-Formel
        const rt60 = 0.161 * volume / (totalArea * meanAlpha);
        
        return rt60;
    }

    // Frühe Reflexionen
    calculateEarlyReflections(dimensions, materials) {
        const reflections = [];
        const source = { x: 0, y: 0, z: 0 }; // Schallquelle
        const receiver = { x: 1, y: 1, z: 1 }; // Empfänger

        // Erste Ordnung Reflexionen
        const surfaces = [
            { normal: [1,0,0], d: 0 }, // Wand 1
            { normal: [-1,0,0], d: dimensions.length }, // Wand 2
            { normal: [0,1,0], d: 0 }, // Wand 3
            { normal: [0,-1,0], d: dimensions.width }, // Wand 4
            { normal: [0,0,1], d: 0 }, // Boden
            { normal: [0,0,-1], d: dimensions.height } // Decke
        ];

        for (const surface of surfaces) {
            const reflection = this.calculateReflectionPoint(
                source, receiver, surface
            );
            if (reflection) {
                reflections.push(reflection);
            }
        }

        return reflections;
    }

    // Hilfsfunktionen
    calculateSurfaces(dimensions) {
        return {
            floor: dimensions.length * dimensions.width,
            ceiling: dimensions.length * dimensions.width,
            wallN: dimensions.length * dimensions.height,
            wallS: dimensions.length * dimensions.height,
            wallE: dimensions.width * dimensions.height,
            wallW: dimensions.width * dimensions.height
        };
    }

    calculateReflectionPoint(source, receiver, surface) {
        // Spiegelquellenmethode
        const d = surface.d;
        const n = surface.normal;
        
        // Spiegelquelle berechnen
        const sourceImage = {
            x: source.x + 2 * (d - (source.x * n[0] + source.y * n[1] + source.z * n[2])) * n[0],
            y: source.y + 2 * (d - (source.x * n[0] + source.y * n[1] + source.z * n[2])) * n[1],
            z: source.z + 2 * (d - (source.x * n[0] + source.y * n[1] + source.z * n[2])) * n[2]
        };

        // Reflexionspunkt berechnen
        const t = ((d - (source.x * n[0] + source.y * n[1] + source.z * n[2])) /
                  ((source.x - receiver.x) * n[0] + (source.y - receiver.y) * n[1] + (source.z - receiver.z) * n[2]));

        const reflectionPoint = {
            x: source.x + t * (receiver.x - source.x),
            y: source.y + t * (receiver.y - source.y),
            z: source.z + t * (receiver.z - source.z)
        };

        return {
            point: reflectionPoint,
            distance: Math.sqrt(
                Math.pow(receiver.x - reflectionPoint.x, 2) +
                Math.pow(receiver.y - reflectionPoint.y, 2) +
                Math.pow(receiver.z - reflectionPoint.z, 2)
            )
        };
    }
}

// Real-time Analysis System
class RealtimeAnalyzer {
    constructor() {
        this.bufferSize = 4096;
        this.analysisInterval = 50; // ms
        this.metrics = {
            latency: 0,
            cpuLoad: 0,
            bufferHealth: 100,
            xruns: 0
        };
    }

    startAnalysis() {
        this.analysisInterval = setInterval(() => {
            this.updateMetrics();
        }, this.analysisInterval);
    }

    stopAnalysis() {
        clearInterval(this.analysisInterval);
    }

    updateMetrics() {
        // Simulierte Metrik-Updates
        this.metrics.latency = Math.random() * 10;
        this.metrics.cpuLoad = Math.random() * 100;
        this.metrics.bufferHealth = 100 - (Math.random() * 20);
        
        // Event auslösen
        this.onMetricsUpdate(this.metrics);
    }

    onMetricsUpdate(metrics) {
        // Update UI elements
        if (document.getElementById('latency-value')) {
            document.getElementById('latency-value').textContent = 
                metrics.latency.toFixed(1) + ' ms';
        }
        if (document.getElementById('cpu-load')) {
            document.getElementById('cpu-load').textContent = 
                metrics.cpuLoad.toFixed(1) + '%';
        }
        if (document.getElementById('buffer-status')) {
            document.getElementById('buffer-status').textContent = 
                metrics.bufferHealth > 80 ? 'Optimal' : 
                metrics.bufferHealth > 60 ? 'Gut' : 
                'Kritisch';
        }
    }
}

// Initialize systems when document is ready
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new AcousticAnalyzer();
    const realtimeAnalyzer = new RealtimeAnalyzer();
    
    // Start real-time analysis
    realtimeAnalyzer.startAnalysis();
    
    // Initialize room simulation with default values
    const defaultRoom = analyzer.simulateRoom(
        { length: 5, width: 4, height: 3 },
        {
            floor: { absorption: 0.1 },
            ceiling: { absorption: 0.05 },
            wallN: { absorption: 0.03 },
            wallS: { absorption: 0.03 },
            wallE: { absorption: 0.03 },
            wallW: { absorption: 0.03 }
        }
    );
    
    console.log('Room simulation results:', defaultRoom);
});