import { describe, expect, it } from "vitest";
import { heatLevel, minutesText, shortDay } from "./Pulse";

describe("pulse words", () => {
  it("reply time reads in minutes, hours or days", () => {
    expect(minutesText(null)).toBe("No replies yet");
    expect(minutesText(42.4)).toBe("42 min");
    expect(minutesText(130)).toBe("2.2 h");
    expect(minutesText(60 * 20)).toBe("20 h");
    expect(minutesText(60 * 72)).toBe("3 days");
  });
  it("the heatmap has an empty step and four steps up to the busiest day", () => {
    expect([0, 1, 5, 6, 10].map((n) => heatLevel(n, 10))).toEqual([0, 1, 2, 3, 4]);
    expect(heatLevel(3, 0)).toBe(0);
  });
  it("days read as short UTC dates", () => {
    expect(shortDay("2026-10-24")).toMatch(/Oct 24|24 Oct/);
  });
});
