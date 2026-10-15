// @vitest-environment jsdom
// Console Refine PR 2 (docs/CONSOLE_UX_AUDIT.md Part 3): the shared
// components behave as the spec says, driven through the DOM.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ConfirmInline } from "./ConfirmInline";
import { Drawer } from "./Drawer";
import { IconButton } from "./IconButton";
import { Meter, meterTone } from "./Meter";
import { Segmented } from "./Segmented";
import { StatusPill } from "./StatusPill";
import { ToastProvider, useToast } from "./Toast";
import { TypedConfirm } from "./TypedConfirm";
import { ErrorBanner } from "./ErrorBanner";
import { ApiError } from "@/lib/api";
import { Trash2 } from "lucide-react";

afterEach(cleanup);

describe("ConfirmInline", () => {
  it("renders the consequence, confirms, cancels, and says when it is permanent", () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();
    render(<ConfirmInline open message="Delete this crystal?" permanent confirmLabel="Delete crystal" onConfirm={onConfirm} onCancel={onCancel} />);
    expect(screen.getByText(/Delete this crystal\?/)).toBeTruthy();
    expect(screen.getByText(/This is permanent\./)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Delete crystal" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
  it("renders nothing while closed", () => {
    const { container } = render(<ConfirmInline open={false} message="x" onConfirm={() => {}} onCancel={() => {}} />);
    expect(container.innerHTML).toBe("");
  });
});

describe("TypedConfirm", () => {
  it("stays disabled until the word matches, and the password when required", () => {
    const onConfirm = vi.fn();
    render(<TypedConfirm word="ERASE" message="Erase every memory." confirmLabel="Erase" requirePassword onConfirm={onConfirm} />);
    const button = screen.getByRole("button", { name: "Erase" }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: "ERASE" } });
    expect(button.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Your password"), { target: { value: "hunter2" } });
    expect(button.disabled).toBe(false);
    fireEvent.click(button);
    expect(onConfirm).toHaveBeenCalledWith("hunter2");
  });
  it("a near miss on the word does not enable the button", () => {
    render(<TypedConfirm word="DELETE" message="m" confirmLabel="Delete" onConfirm={() => {}} />);
    fireEvent.change(screen.getByLabelText(/Type/), { target: { value: "delete" } });
    expect((screen.getByRole("button", { name: "Delete" }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("Segmented", () => {
  it("arrow keys move the selection and the group is a radiogroup", () => {
    const onChange = vi.fn();
    render(<Segmented label="Group by" value="none" onChange={onChange}
      options={[{ value: "none", label: "None" }, { value: "tier", label: "Tier" }, { value: "type", label: "Type" }]} />);
    expect(screen.getByRole("radiogroup", { name: "Group by" })).toBeTruthy();
    const none = screen.getByRole("radio", { name: "None" });
    expect(none.getAttribute("aria-checked")).toBe("true");
    fireEvent.keyDown(none, { key: "ArrowRight" });
    expect(onChange).toHaveBeenCalledWith("tier");
    fireEvent.keyDown(none, { key: "ArrowLeft" });
    expect(onChange).toHaveBeenCalledWith("type");
  });
});

describe("StatusPill", () => {
  it("shows a word and keeps the raw enum in title", () => {
    render(<StatusPill variant="ok" raw="whitelist">Trusted</StatusPill>);
    const el = screen.getByText("Trusted");
    expect(el.getAttribute("title")).toBe("whitelist");
    expect(el.className).toContain("text-ok");
  });
});

describe("Meter", () => {
  it("colors by threshold: accent under 85, warn to 99, bad at 100", () => {
    expect(meterTone(0)).toBe("accent");
    expect(meterTone(84.9)).toBe("accent");
    expect(meterTone(85)).toBe("warn");
    expect(meterTone(99.9)).toBe("warn");
    expect(meterTone(100)).toBe("bad");
    render(<Meter label="Crystal facts" value={90} max={100} valueLabel="90 / 100" />);
    const bar = screen.getByRole("progressbar", { name: "Crystal facts" });
    expect(bar.getAttribute("aria-valuenow")).toBe("90");
    expect(bar.getAttribute("data-tone")).toBe("warn");
  });
});

describe("IconButton", () => {
  it("carries its label as aria-label, title and a visible tooltip", () => {
    render(<IconButton label="Delete fact" icon={Trash2} />);
    const btn = screen.getByRole("button", { name: "Delete fact" });
    expect(btn.getAttribute("title")).toBe("Delete fact");
    expect(screen.getByRole("tooltip", { hidden: true }).textContent).toBe("Delete fact");
  });
});

describe("ErrorBanner", () => {
  it("shows the server's message from an error envelope and offers Retry only when given one", () => {
    const envelope = new ApiError(400, "Bad Request", { error: { message: "That file is too large." } });
    const onRetry = vi.fn();
    render(<ErrorBanner title="Upload" error={envelope} onRetry={onRetry} />);
    expect(screen.getByRole("alert").textContent).toContain("That file is too large.");
    fireEvent.click(screen.getByRole("button", { name: /Retry/ }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    cleanup();
    render(<ErrorBanner message="Plain" />);
    expect(screen.queryByRole("button", { name: /Retry/ })).toBeNull();
  });
});

describe("Drawer", () => {
  it("is a labelled dialog that closes on Escape and gives focus back to the opener", () => {
    const onClose = vi.fn();
    function Host({ open }: { open: boolean }) {
      return (
        <>
          <button>Open</button>
          <Drawer open={open} onClose={onClose} title="Crystal">
            <input aria-label="inside" />
          </Drawer>
        </>
      );
    }
    const { rerender } = render(<Host open={false} />);
    const opener = screen.getByRole("button", { name: "Open" });
    opener.focus();
    rerender(<Host open />);
    const dialog = screen.getByRole("dialog", { name: "Crystal" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    expect(document.activeElement && dialog.contains(document.activeElement)).toBe(true);
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    rerender(<Host open={false} />);
    expect(document.activeElement).toBe(opener);
  });
});

describe("Toast", () => {
  it("announces through role=status, runs the Undo action, and hides on its own", () => {
    vi.useFakeTimers();
    const undo = vi.fn();
    function Page() {
      const toast = useToast();
      return <button onClick={() => toast("Fact retired", { action: { label: "Undo", onClick: undo } })}>Retire</button>;
    }
    render(<ToastProvider><Page /></ToastProvider>);
    fireEvent.click(screen.getByRole("button", { name: "Retire" }));
    expect(screen.getByRole("status").textContent).toContain("Fact retired");
    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
    expect(undo).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("status")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retire" }));
    act(() => { vi.advanceTimersByTime(2600); });
    expect(screen.queryByRole("status")).toBeNull();
    vi.useRealTimers();
  });
});
