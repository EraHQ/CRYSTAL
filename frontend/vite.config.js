import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
// Crystal Cache Inspector — Vite config.
//
// Important:
//   - `base: "/admin/"` so the SPA is correctly served from FastAPI's
//     StaticFiles mount at /admin (see app.py).
//   - The dev server proxies /admin/api/* and /v1/* to localhost:8000
//     so `npm run dev` works without CORS surgery.
export default defineConfig({
    plugins: [react()],
    base: "/admin/",
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
    server: {
        port: 5173,
        proxy: {
            "/admin/api": "http://localhost:8000",
            "/v1": "http://localhost:8000",
        },
    },
    build: {
        outDir: "dist",
        emptyOutDir: true,
        sourcemap: true,
    },
});
