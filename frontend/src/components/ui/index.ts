// Shared console components (docs/CONSOLE_UX_AUDIT.md Part 3). Every
// screen imports from "@/components/ui"; the pre-Refine pieces live in
// ./legacy until the screens that use them are rebuilt.
export { PageHeader } from "./PageHeader";
export { Toolbar } from "./Toolbar";
export { FilterChips } from "./FilterChips";
export type { FilterChipOption } from "./FilterChips";
export { Segmented } from "./Segmented";
export type { SegmentedOption } from "./Segmented";
export { StatusPill } from "./StatusPill";
export type { StatusVariant } from "./StatusPill";
export { EmptyState } from "./EmptyState";
export { ErrorBanner } from "./ErrorBanner";
export { ConfirmInline } from "./ConfirmInline";
export { TypedConfirm } from "./TypedConfirm";
export { Drawer } from "./Drawer";
export { Modal } from "./Modal";
export { ToastProvider, useToast } from "./Toast";
export type { ToastOptions } from "./Toast";
export { Meter, meterTone } from "./Meter";
export { IconButton } from "./IconButton";
export { Skeleton } from "./Skeleton";
export { Disclosure, JsonView, MatchPill, QualityPill, LoadingRows, CrystalButton, TypeBadge } from "./legacy";
