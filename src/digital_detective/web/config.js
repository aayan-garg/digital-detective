/**
 * Digital Detective — Web Client Configuration
 *
 * For decoupled public deployment (e.g. Vercel frontend calling Render backend):
 * Set window.__DIGITAL_DETECTIVE_API_BASE__ to your Render public service URL.
 *
 * Example:
 *   window.__DIGITAL_DETECTIVE_API_BASE__ = "https://digital-detective-api.onrender.com";
 *
 * Precedence Order in app.js:
 * 1. URL Query Parameter (?api=https://...)
 * 2. LocalStorage ("DIGITAL_DETECTIVE_API_BASE")
 * 3. window.__DIGITAL_DETECTIVE_API_BASE__ (this file)
 * 4. window.location.origin (unified hosting fallback)
 */

window.__DIGITAL_DETECTIVE_API_BASE__ = "";
