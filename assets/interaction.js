// Main interaction handler
class AcousticInteraction {
    constructor() {
        this.initializeVisualization();
        this.setupEventListeners();
        this.setupCharts();
    }

    initializeVisualization() {
        this.viz = new WaveformVisualizer('waveform-viz');
        this.isPlaying = false;
    }

    setupEventListeners() {
        // Play/Pause button
        const playPauseBtn = document.getElementById('play-pause');
        if (playPauseBtn) {
            playPauseBtn.addEventListener('click', () => {
                this.isPlaying = !this.isPlaying;
                this.viz.toggleAnimation(this.isPlaying);
            });
        }

        // Wave change button
        const changeWaveBtn = document.getElementById('change-wave');
        if (changeWaveBtn) {
            changeWaveBtn.addEventListener('click', () => {
                this.viz.changeWaveform();
            });
        }

        // Language switcher
        document.querySelectorAll('.language-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.switchLanguage(e.target.dataset.lang);
            });
        });
    }

    setupCharts() {
        // Set up frequency analysis chart
        const freqCtx = document.getElementById('frequency-chart');
        if (freqCtx) {
            this.frequencyChart = new FrequencyAnalyzer(freqCtx);
        }
    }

    switchLanguage(lang) {
        document.querySelectorAll('[data-lang]').forEach(elem => {
            elem.style.display = elem.dataset.lang === lang ? 'block' : 'none';
        });
    }
}

// Waveform visualization
class WaveformVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.waves = [
            { amplitude: 30, frequency: 0.02, speed: 0.02, color: '#4A90E2' },
            { amplitude: 20, frequency: 0.03, speed: 0.03, color: '#67B26F' },
            { amplitude: 15, frequency: 0.01, speed: 0.01, color: '#FF6B6B' }
        ];
        
        this.time = 0;
        this.isAnimating = false;
        this.setupCanvas();
    }

    setupCanvas() {
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    toggleAnimation(start) {
        this.isAnimating = start;
        if (start) {
            this.animate();
        }
    }

    changeWaveform() {
        this.waves = this.waves.map(wave => ({
            ...wave,
            amplitude: Math.random() * 30 + 10,
            frequency: Math.random() * 0.03 + 0.01
        }));
    }

    animate() {
        if (!this.isAnimating) return;

        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.waves.forEach(wave => {
            this.ctx.beginPath();
            this.ctx.moveTo(0, this.canvas.height / 2);
            
            for (let x = 0; x < this.canvas.width; x++) {
                const y = this.canvas.height / 2 + 
                         wave.amplitude * 
                         Math.sin(x * wave.frequency + this.time * wave.speed);
                this.ctx.lineTo(x, y);
            }
            
            this.ctx.strokeStyle = wave.color;
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
        });
        
        this.time += 0.05;
        requestAnimationFrame(() => this.animate());
    }
}

// Frequency analysis visualization
class FrequencyAnalyzer {
    constructor(canvas) {
        this.ctx = canvas.getContext('2d');
        this.setupChart();
    }

    setupChart() {
        const frequencies = Array.from({length: 100}, (_, i) => i * 48000 / 100);
        const magnitudes = this.generateRandomMagnitudes(100);
        
        new Chart(this.ctx, {
            type: 'line',
            data: {
                labels: frequencies,
                datasets: [{
                    label: 'Frequency Response',
                    data: magnitudes,
                    borderColor: '#4A90E2',
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Frequency (Hz)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Magnitude (dB)'
                        }
                    }
                }
            }
        });
    }

    generateRandomMagnitudes(count) {
        return Array.from({length: count}, () => Math.random() * 40 - 20);
    }
}

// Initialize everything when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AcousticInteraction();
});