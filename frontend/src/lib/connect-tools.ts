// T2b (2026-09-26): the connect-tool registry. ONE entry per tool the
// environment picker offers; adding a tool = adding an entry, never a
// redesign. Every MCP client shares the same server URL + Bearer shape,
// so entries differ only in where the config goes. Ratified 2026-09-25:
// any MCP client, never Claude-only.
//
// LIVE-FOUND 2026-09-24: Claude Desktop and Claude.ai's "Add custom
// connector" dialog is the OAUTH route (it probes sign-in discovery and
// fails against a static-key server until L2-S5 ships). The static-key
// route that works TODAY is the claude_desktop_config.json file. Every
// entry now carries explicit numbered `steps`, rendered prominently by
// the wizard — the paste location was previously a truncated caption
// and users walked straight into the OAuth dialog.
export const MCP_URL =
  "https://crystal-api-118881845105.us-east5.run.app/mcp";

const jsonBlock = (key: string) => `{
  "mcpServers": {
    "crystal-cache": {
      "type": "http",
      "url": "${MCP_URL}",
      "headers": { "Authorization": "Bearer ${key}" }
    }
  }
}`;

export interface ConnectTool {
  id: string;
  label: string;
  // Explicit, numbered, do-this-then-that. Rendered large by the wizard.
  steps: string[];
  snippet: (key: string) => string;
}

export const CONNECT_TOOLS: ConnectTool[] = [
  {
    id: "claude_desktop", label: "Claude Desktop",
    steps: [
      "Open Settings > Connectors and click Add custom connector",
      "Name it Crystal, paste the URL below, and click Add",
      "On the options screen keep what Claude detected: Sign in now, Register automatically — no headers needed",
      "Click Connect and sign in with your Crystal account",
      "Ask Claude: \"what do you remember about me?\" — this screen flips to Connected the moment it answers",
    ],
    snippet: () => MCP_URL,
  },
  {
    id: "claude_code", label: "Claude Code",
    steps: [
      "Open any terminal",
      "Run the command below",
      "Start (or restart) a Claude Code session and ask: \"what do you remember about me?\"",
    ],
    snippet: (k) =>
      `claude mcp add crystal-cache ${MCP_URL} -t http -H "Authorization: Bearer ${k}"`,
  },
  {
    id: "claude_ai", label: "Claude.ai (web)",
    steps: [
      "On claude.ai open Settings > Connectors and click Add custom connector",
      "Name it Crystal, paste the URL below, and click Add",
      "On the options screen keep what Claude detected: Sign in now, Register automatically — no headers needed",
      "Click Connect and sign in with your Crystal account",
      "Ask Claude: \"what do you remember about me?\" — this screen flips to Connected the moment it answers",
    ],
    snippet: () => MCP_URL,
  },
  {
    id: "cursor", label: "Cursor",
    steps: [
      "Open ~/.cursor/mcp.json in any text editor (or Cursor Settings > MCP > Add new global MCP server)",
      "Paste the snippet below",
      "Restart Cursor, then ask its agent: \"what do you remember about me?\"",
    ],
    snippet: jsonBlock,
  },
  {
    id: "windsurf", label: "Windsurf",
    steps: [
      "Open ~/.codeium/windsurf/mcp_config.json (or Settings > Cascade > MCP servers > Add)",
      "Paste the snippet below",
      "Restart Windsurf, then ask Cascade: \"what do you remember about me?\"",
    ],
    snippet: jsonBlock,
  },
  {
    id: "vscode_copilot", label: "VS Code (Copilot)",
    steps: [
      "Open the command palette (Ctrl/Cmd+Shift+P) and run: MCP: Add Server",
      "Choose HTTP, paste the URL from the snippet, and add the Authorization header shown",
      "Or paste the whole snippet into .vscode/mcp.json, then reload the window",
    ],
    snippet: (k) => `{
  "servers": {
    "crystal-cache": {
      "type": "http",
      "url": "${MCP_URL}",
      "headers": { "Authorization": "Bearer ${k}" }
    }
  }
}`,
  },
  {
    id: "cline", label: "Cline",
    steps: [
      "Open Cline's MCP Servers panel and click Configure MCP Servers",
      "Paste the snippet below into cline_mcp_settings.json",
      "Save — Cline reloads servers automatically",
    ],
    snippet: jsonBlock,
  },
  {
    id: "zed", label: "Zed",
    steps: [
      "Open Zed's settings.json (Cmd/Ctrl+, then the JSON view)",
      "Add the context_servers block below",
      "Restart Zed",
    ],
    snippet: (k) => `"context_servers": {
  "crystal-cache": {
    "url": "${MCP_URL}",
    "headers": { "Authorization": "Bearer ${k}" }
  }
}`,
  },
  {
    id: "chatgpt", label: "ChatGPT",
    steps: [
      "Open ChatGPT Settings > Connectors and enable Developer mode if you haven't",
      "Click Add connector and paste the server URL and the Authorization header from the snippet",
      "Save, then ask ChatGPT: \"what do you remember about me?\"",
    ],
    snippet: (k) => `Server URL: ${MCP_URL}\nHeader:  Authorization: Bearer ${k}`,
  },
  {
    id: "gemini_cli", label: "Gemini CLI",
    steps: [
      "Open ~/.gemini/settings.json in any text editor",
      "Add the mcpServers block below",
      "Restart the Gemini CLI session",
    ],
    snippet: (k) => `"mcpServers": {
  "crystal-cache": {
    "httpUrl": "${MCP_URL}",
    "headers": { "Authorization": "Bearer ${k}" }
  }
}`,
  },
  {
    id: "other_mcp", label: "Other MCP client",
    steps: [
      "Find where your client adds an MCP server (streamable HTTP)",
      "Give it the URL and the Authorization header below — that is the entire contract",
    ],
    snippet: (k) => `Server URL: ${MCP_URL}\nAuth header: Authorization: Bearer ${k}`,
  },
  {
    id: "direct_api", label: "Direct API",
    steps: [
      "The same key works on the REST surface — try it now:",
    ],
    snippet: (k) =>
      `curl -H "Authorization: Bearer ${k}" \\\n  https://crystal-api-118881845105.us-east5.run.app/v1/me`,
  },
];
