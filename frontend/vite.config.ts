import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Dev-only proxy so the browser talks to one origin and we avoid CORS
    // during development. In production VITE_API_BASE_URL points at the
    // deployed API instead.
    proxy: {
      "/health": "http://localhost:8000",
      "/api": "http://localhost:8000",
    },
  },
});
