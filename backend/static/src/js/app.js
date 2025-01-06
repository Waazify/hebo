import Alpine from 'alpinejs'
import htmx from 'htmx.org'

// Make Alpine available globally
window.Alpine = Alpine
Alpine.start()

// Custom Alpine components
Alpine.data('dropdown', () => ({
    open: false,
    toggle() {
        this.open = !this.open
    }
})) 