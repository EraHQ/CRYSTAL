// Selected-customer state. Persists in localStorage so a refresh keeps
// the same customer picked. Three pages read from this context; each
// page also handles the "no customer selected" empty state.
import { createContext, useContext, useEffect, useState, ReactNode } from "react";

interface SelectedCustomerContextShape {
  selectedCustomerId: string | null;
  setSelectedCustomerId: (id: string | null) => void;
}

const SelectedCustomerContext =
  createContext<SelectedCustomerContextShape | null>(null);

const STORAGE_KEY = "crystal-cache-inspector.selected-customer-id";

export function SelectedCustomerProvider({ children }: { children: ReactNode }) {
  const [selectedCustomerId, _setSelectedCustomerId] = useState<string | null>(
    () => {
      try {
        return localStorage.getItem(STORAGE_KEY);
      } catch {
        return null;
      }
    }
  );

  const setSelectedCustomerId = (id: string | null) => {
    _setSelectedCustomerId(id);
    try {
      if (id === null) localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, id);
    } catch {
      // localStorage can be unavailable (incognito with strict mode).
      // Proceeding with in-memory state only is fine.
    }
  };

  // If something elsewhere clears the storage, sync.
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        _setSelectedCustomerId(e.newValue);
      }
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  return (
    <SelectedCustomerContext.Provider
      value={{ selectedCustomerId, setSelectedCustomerId }}
    >
      {children}
    </SelectedCustomerContext.Provider>
  );
}

export function useSelectedCustomer() {
  const ctx = useContext(SelectedCustomerContext);
  if (ctx === null) {
    throw new Error(
      "useSelectedCustomer must be used inside SelectedCustomerProvider"
    );
  }
  return ctx;
}
