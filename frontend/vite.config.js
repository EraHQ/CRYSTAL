import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
// Crystal Cache Inspector — Vite config.
//
// Important:
//   - `base: "/admin/"` so the SPA is correctly served from FastAPI's
//     StaticFiles mount at /admin (see app.py).
//   - The dev server proxies /admin/api/* and /v1/* to the API so
//     `npm run dev` works without CORS surgery. The target defaults to
//     a local server; point it anywhere (e.g. the hosted deployment)
//     with VITE_API_TARGET:
//       VITE_API_TARGET=https://crystal-api-XXXX.run.app npm run dev
export default defineConfig(function () {
    var apiTarget = process.env.VITE_API_TARGET || "http://localhost:8000";
    return {
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
                "/admin/api": { target: apiTarget, changeOrigin: true },
                "/v1": { target: apiTarget, changeOrigin: true },
            },
        },
        build: {
            outDir: "dist",
            emptyOutDir: true,
            sourcemap: true,
        },
    };
});
