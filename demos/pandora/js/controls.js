/**
 * Interactive controls: mouse, keyboard, UI elements.
 */

export class Controls {
    constructor(canvas, renderer, callbacks) {
        this.canvas = canvas;
        this.renderer = renderer;
        this.callbacks = callbacks; // { onPause, onReset, onRandomize, onDtChange, onStepsChange, onParticleCountChange, onSpeciesCountChange }
        this.isDragging = false;
        this.lastMouse = [0, 0];

        this._setupMouse();
        this._setupKeyboard();
    }

    _setupMouse() {
        const canvas = this.canvas;

        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            this.renderer.zoom *= factor;
        }, { passive: false });

        canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouse = [e.clientX, e.clientY];
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            const dx = e.clientX - this.lastMouse[0];
            const dy = e.clientY - this.lastMouse[1];
            this.lastMouse = [e.clientX, e.clientY];

            // Convert pixel delta to world delta
            const scale = this.renderer.boxSize / (this.renderer.zoom * Math.min(canvas.width, canvas.height));
            this.renderer.pan[0] -= dx * scale;
            this.renderer.pan[1] += dy * scale; // Y is flipped
        });

        canvas.addEventListener('mouseup', () => { this.isDragging = false; });
        canvas.addEventListener('mouseleave', () => { this.isDragging = false; });
    }

    _setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            switch (e.code) {
                case 'Space':
                    e.preventDefault();
                    this.callbacks.onPause?.();
                    break;
                case 'KeyR':
                    this.callbacks.onReset?.();
                    break;
                case 'KeyN':
                    this.callbacks.onRandomize?.();
                    break;
            }
        });
    }
}
