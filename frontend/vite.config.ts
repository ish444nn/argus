import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Where the dev server forwards API calls. `localhost` when running
// `npm run dev` on the host; the Compose service name when running in the
// frontend container, which cannot see the host's localhost.
const API_TARGET = process.env.VITE_DEV_API_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Proxy so the browser talks to a single origin and CORS never enters the
    // picture. Set VITE_API_BASE_URL to bypass it and call the API directly.
    proxy: {
      "/health": API_TARGET,
      "/api": API_TARGET,
    },
    // Needed for hot reload to reach the browser from inside a container.
    watch: { usePolling: true },
  },
});
