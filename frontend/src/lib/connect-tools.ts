// T2b (2026-09-26): the connect-tool registry. ONE entry per tool the
// environment picker offers; adding a tool = adding an entry, never a
// redesign. Every MCP client shares the same server URL + Bearer shape,
// so entries differ only in where the config goes. Ratified 2026-09-25:
// any MCP client, never Claude-only.
export const MCP_URL =
  "https://crystal-api-118881845105.us-east5.run.app/mcp";

const jsonBlock = (key: string) => `{
  "mcpServers": {
    "crystal-cache": {
      "url": "${MCP_URL}",
      "headers": { "Authorization": "Bearer ${key}" }
    }
  }
}`;

export interface ConnectTool {
  id: string;
  label: string;
  pasteLine: string;
  snippet: (key: string) => string;
}

export const CONNECT_TOOLS: ConnectTool[] = [
  {
    id: "claude_desktop", label: "Claude Desktop",
    pasteLine: "Settings > Connectors > Add custom connector, or claude_desktop_config.json",
    snippet: jsonBlock,
  },
  {
    id: "claude_code", label: "Claude Code",
    pasteLine: "One command in any terminal",
    snippet: (k) =>
      `claude mcp add crystal-cache ${MCP_URL} -t http -H "Authorization: Bearer ${k}"`,
  },
  {
    id: "claude_ai", label: "Claude.ai (web)",
    pasteLine: "Settings > Connectors (one-click connect is coming; until then use Desktop or Code)",
    snippet: () =>
      "Claude.ai's connector flow requires OAuth, which ships shortly.\nYour key already works everywhere below in the meantime.",
  },
  {
    id: "cursor", label: "Cursor",
    pasteLine: "Cursor Settings > MCP > Add new server (~/.cursor/mcp.json)",
    snippet: jsonBlock,
  },
  {
    id: "windsurf", label: "Windsurf",
    pasteLine: "Settings > Cascade > MCP servers (~/.codeium/windsurf/mcp_config.json)",
    snippet: jsonBlock,
  },
  {
    id: "vscode_copilot", label: "VS Code (Copilot)",
    pasteLine: "Command palette: MCP: Add Server, or .vscode/mcp.json",
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
    pasteLine: "MCP Servers panel > Configure (cline_mcp_settings.json)",
    snippet: jsonBlock,
  },
  {
    id: "zed", label: "Zed",
    pasteLine: "settings.json > context_servers",
    snippet: (k) => `"context_servers": {
  "crystal-cache": {
    "url": "${MCP_URL}",
    "headers": { "Authorization": "Bearer ${k}" }
  }
}`,
  },
  {
    id: "chatgpt", label: "ChatGPT",
    pasteLine: "Settings > Connectors > Add (developer mode) — paste the URL and header",
    snippet: (k) => `Server URL: ${MCP_URL}\nHeader:  Authorization: Bearer ${k}`,
  },
  {
    id: "gemini_cli", label: "Gemini CLI",
    pasteLine: "~/.gemini/settings.json > mcpServers",
    snippet: (k) => `"mcpServers": {
  "crystal-cache": {
    "httpUrl": "${MCP_URL}",
    "headers": { "Authorization": "Bearer ${k}" }
  }
}`,
  },
  {
    id: "other_mcp", label: "Other MCP client",
    pasteLine: "Any MCP client that speaks streamable HTTP",
    snippet: (k) => `Server URL: ${MCP_URL}\nAuth header: Authorization: Bearer ${k}`,
  },
  {
    id: "direct_api", label: "Direct API",
    pasteLine: "Straight HTTP — the same key works on the REST surface",
    snippet: (k) =>
      `curl -H "Authorization: Bearer ${k}" \\\n  https://crystal-api-118881845105.us-east5.run.app/v1/me`,
  },
];
