// Nango's Connect UI (2026-10-24). The same protocol @nangohq/frontend
// 0.71.13's ConnectUI speaks, in ~50 lines so the console takes no new
// dependency: a full-screen iframe on connect.nango.dev; when it posts
// "ready" we post back the session token our server minted; it posts
// "connect" when a sign-in finished and "close" when the person leaves.
// The CSP's frame-src allows https://connect.nango.dev
// (deploy/inspector/nginx.conf.template).
export const CONNECT_ORIGIN = "https://connect.nango.dev";

export interface ConnectEvent {
  type: string;
  payload?: Record<string, unknown>;
}

/** Opens the Connect UI; returns a function that closes it. */
export function openConnectUI(sessionToken: string, onEvent: (e: ConnectEvent) => void, doc: Document = document): () => void {
  const url = new URL(CONNECT_ORIGIN + "/");
  url.searchParams.append("apiURL", "https://api.nango.dev");
  url.searchParams.append("detectClosedAuthWindow", "true");
  const iframe = doc.createElement("iframe");
  iframe.src = url.href;
  iframe.id = "connect-ui";
  iframe.title = "Connect an app";
  Object.assign(iframe.style, {
    position: "fixed", zIndex: "9999", top: "0", left: "0", right: "0", bottom: "0",
    width: "100vw", height: "100vh", border: "0", backgroundColor: "transparent",
  });
  iframe.allow = "clipboard-write";
  const win = doc.defaultView ?? window;
  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    win.removeEventListener("message", listener);
    iframe.remove();
    doc.body.style.overflow = "";
  };
  const listener = (event: MessageEvent) => {
    if (event.origin !== CONNECT_ORIGIN) return;
    const data = event.data as ConnectEvent | null;
    if (!data || typeof data !== "object" || typeof data.type !== "string") return;
    if (data.type === "ready") {
      iframe.contentWindow?.postMessage({ type: "session_token", sessionToken }, "*");
    }
    if (data.type === "close") close();
    onEvent(data);
  };
  win.addEventListener("message", listener, false);
  doc.body.append(iframe);
  doc.body.style.overflow = "hidden";
  return close;
}
