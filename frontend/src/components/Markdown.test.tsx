// @vitest-environment jsdom
// Markdown-lite: what the models emit renders as structure, never raw
// markers; only http(s) links become links.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Markdown, parseBlocks } from "./Markdown";

afterEach(cleanup);

describe("parseBlocks", () => {
  it("reads headings, lists, tables, rules and paragraphs", () => {
    const blocks = parseBlocks("# Title\n\nFirst line\nsecond line\n\n- one\n- two\n1. a\n2) b\n\n| h1 | h2 |\n|---|:--:|\n| x | y |\n\n---\n\nLast<br>break");
    expect(blocks.map((b) => b.kind)).toEqual(["h", "p", "ul", "ol", "table", "hr", "p"]);
    expect(blocks[1]).toEqual({ kind: "p", text: "First line\nsecond line" });
    expect(blocks[2]).toEqual({ kind: "ul", items: ["one", "two"] });
    expect(blocks[3]).toEqual({ kind: "ol", items: ["a", "b"], start: 1 });
    expect(blocks[4]).toEqual({ kind: "table", rows: [["h1", "h2"], ["x", "y"]] });
    expect(blocks[6]).toEqual({ kind: "p", text: "Last\nbreak" });
  });
});

describe("Markdown", () => {
  it("renders structure and safe links, keeps code blocks and bold", () => {
    render(<Markdown text={"### Why\n\nSee **this** and [docs](https://example.com/x) and [bad](javascript:alert(1)).\n\n```json\n{\"a\":1}\n```"} />);
    expect(screen.getByRole("heading", { level: 3 }).textContent).toBe("Why");
    expect(screen.getByText("this").tagName).toBe("STRONG");
    const link = screen.getByRole("link", { name: "docs" });
    expect(link.getAttribute("href")).toBe("https://example.com/x");
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(screen.queryByRole("link", { name: "bad" })).toBeNull();
    expect(screen.getByText(/\[bad\]\(javascript:alert\(1\)\)/)).toBeTruthy();
    expect(screen.getByText("json")).toBeTruthy();
    expect(screen.getByText('{"a":1}').tagName).toBe("PRE");
    expect(document.body.textContent).not.toContain("###");
  });
  it("renders a pipe table as a table", () => {
    render(<Markdown text={"| Model | Price |\n|---|---|\n| Super | $0.30 |"} />);
    expect(screen.getByRole("table")).toBeTruthy();
    expect(screen.getByRole("columnheader", { name: "Model" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "$0.30" })).toBeTruthy();
  });
});
