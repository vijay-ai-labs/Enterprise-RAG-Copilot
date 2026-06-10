import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:8000",
      "/ingest": "http://localhost:8000",
      "/query": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/feedback": "http://localhost:8000",
      "/documents": "http://localhost:8000",
      "/conversations": "http://localhost:8000",
      "/cache": "http://localhost:8000",
    },
  },
});
