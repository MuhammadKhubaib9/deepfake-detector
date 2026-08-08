import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Build output = frontend/dist, served by Flask (python app.py) as the app UI.
// Dev mode: `npm run dev` (port 5173), proxying /api + /media to Flask :5000.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/media": "http://127.0.0.1:5000"
    }
  },
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});