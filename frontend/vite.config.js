import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Backend the dev proxy forwards to. Override with e.g.
// API_TARGET=http://127.0.0.1:8001 npm run dev
const apiTarget = process.env.API_TARGET || 'http://120.138.7.130:8001'
// Multi-camera live view (app/web/server.py). Override with CAMERAS_TARGET.
const camerasTarget = process.env.CAMERAS_TARGET || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Must come before '/api' -- the first matching key wins.
      '/api/cameras': {
        target: camerasTarget,
        changeOrigin: true,
      },
      // Dev-time only: forwards /api/* to the FastAPI backend so the
      // browser never needs to know its port, and there's no CORS to deal
      // with. In production, serve the built frontend from behind the same
      // reverse proxy as the API instead.
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      // Enrollment/attendance thumbnails served by the attendance backend.
      '/data': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/ws': {
        target: apiTarget.replace(/^http/, 'ws'),
        ws: true,
      },
    },
  },
})
