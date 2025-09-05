// Interactive Acoustic Visualization
class AcousticViz {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.createElement('canvas');
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        
        this.setup();
        this.animate();
    }

    setup() {
        this.canvas.width = this.container.clientWidth;
        this.canvas.height = 400;
        
        // Responsive handling
        window.addEventListener('resize', () => {
            this.canvas.width = this.container.clientWidth;
            this.canvas.height = 400;
        });

        // Initialize wave parameters
        this.waves = [
            { amplitude: 30, frequency: 0.02, speed: 0.02, color: '#4A90E2' },
            { amplitude: 20, frequency: 0.03, speed: 0.03, color: '#67B26F' },
            { amplitude: 15, frequency: 0.01, speed: 0.01, color: '#FF6B6B' }
        ];
        
        this.time = 0;
    }

    drawWave(wave) {
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
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw each wave
        this.waves.forEach(wave => this.drawWave(wave));
        
        this.time += 0.05;
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize visualizations when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create English visualization
    new AcousticViz('interactive-demos');
    
    // Create German visualization
    new AcousticViz('interaktive-demos');
});