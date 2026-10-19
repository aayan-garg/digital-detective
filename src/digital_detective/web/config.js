/**
 * Digital Detective — Web Client Configuration
 *
 * Public Deployment: Self-contained zero-cost interactive static demo.
 * Serves verified benchmark investigations locally from demo_investigations.json
 * without requiring any backend hosting, external API, or cloud infrastructure.
 *
 * For optional local FastAPI execution:
 * Start the local backend:
 *   $env:RCAEVAL_DATASET_ROOT="data/demo_cases"
 *   uvicorn digital_detective.api.main:app --host 0.0.0.0 --port 8000
 * Then access: http://localhost:8000/
 */

// Empty by default for zero-cost static demo operation
window.__DIGITAL_DETECTIVE_API_BASE__ = "";

// Explicit flag indicating static benchmark demo mode
window.__DIGITAL_DETECTIVE_STATIC_DEMO__ = true;
