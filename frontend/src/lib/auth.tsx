// Hosted-auth layer (Accounts Phase C, 2026-07-06).
//
// Presence-as-switch, mirrored from the backend's D4: the login surface
// exists ONLY when the Vite build carries VITE_FIREBASE_API_KEY /
// _AUTH_DOMAIN / _PROJECT_ID. A self-host build without them renders the
// console exactly as before — no login, no Firebase in the runtime path.
//
// Session flow: Firebase SDK owns the session; every API call gains the
// ID token via api.setAuthTokenProvider; /v1/me pins role + tenant; a
// valid session with NO account (401 from /v1/me) routes to onboarding,
// whose signup call provisions the managed tenant.
import {
  createContext,
  useContext,
  useEffect,
  useState,
  ReactNode,
} from "react";
import { initializeApp, type FirebaseApp } from "firebase/app";
import {
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  getAuth,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut as fbSignOut,
  type Auth,
  type User,
} from "firebase/auth";
import { api, setAuthTokenProvider } from "./api";

export interface Me {
  kind: string;
  role: string;
  customer_id: string | null;
  user_id: string | null;
  email: string | null;
}

export type AuthStatus =
  | "disabled"     // no Firebase config: self-host, console as-is
  | "loading"      // Firebase resolving the session
  | "signedOut"    // hosted mode, no session → Login page
  | "needsSignup"  // session valid, no account → Onboarding page
  | "signedIn";    // session + account → console

interface AuthShape {
  status: AuthStatus;
  me: Me | null;
  email: string | null;
  error: string | null;
  signInGoogle: () => Promise<void>;
  signInEmail: (email: string, password: string) => Promise<void>;
  signUpEmail: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refreshMe: () => Promise<void>;
}

const AuthContext = createContext<AuthShape | null>(null);

export function firebaseConfigured(): boolean {
  return Boolean(
    import.meta.env.VITE_FIREBASE_API_KEY &&
      import.meta.env.VITE_FIREBASE_AUTH_DOMAIN &&
      import.meta.env.VITE_FIREBASE_PROJECT_ID
  );
}

let app: FirebaseApp | null = null;
function auth(): Auth {
  if (!app) {
    app = initializeApp({
      apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
      authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
      projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
    });
  }
  return getAuth(app);
}

function friendly(e: unknown): string {
  const code = (e as { code?: string })?.code ?? "";
  if (code.includes("invalid-credential") || code.includes("wrong-password"))
    return "Email or password is incorrect.";
  if (code.includes("user-not-found")) return "No account with that email.";
  if (code.includes("email-already-in-use"))
    return "That email already has an account — sign in instead.";
  if (code.includes("weak-password"))
    return "Password needs at least 6 characters.";
  if (code.includes("popup-closed")) return "Sign-in was cancelled.";
  return "Sign-in failed. Please try again.";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const enabled = firebaseConfigured();
  const [status, setStatus] = useState<AuthStatus>(
    enabled ? "loading" : "disabled"
  );
  const [user, setUser] = useState<User | null>(null);
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Resolve /v1/me for a signed-in Firebase user. 401 = valid identity,
  // no account yet → the onboarding/signup path.
  const resolveMe = async (u: User) => {
    setAuthTokenProvider(() => u.getIdToken());
    try {
      const m = await api.me();
      setMe(m);
      setStatus("signedIn");
    } catch (e) {
      if ((e as { status?: number })?.status === 401) {
        setMe(null);
        setStatus("needsSignup");
      } else {
        setError("Could not reach the server. Retrying may help.");
        setStatus("signedOut");
      }
    }
  };

  useEffect(() => {
    if (!enabled) return;
    const unsub = onAuthStateChanged(auth(), (u) => {
      setUser(u);
      setError(null);
      if (!u) {
        setAuthTokenProvider(null);
        setMe(null);
        setStatus("signedOut");
      } else {
        void resolveMe(u);
      }
    });
    return unsub;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  const wrap = (fn: () => Promise<unknown>) => async () => {
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(friendly(e));
    }
  };

  const shape: AuthShape = {
    status,
    me,
    email: user?.email ?? null,
    error,
    signInGoogle: wrap(() =>
      signInWithPopup(auth(), new GoogleAuthProvider())
    ),
    signInEmail: async (email, password) => {
      setError(null);
      try {
        await signInWithEmailAndPassword(auth(), email, password);
      } catch (e) {
        setError(friendly(e));
      }
    },
    signUpEmail: async (email, password) => {
      setError(null);
      try {
        await createUserWithEmailAndPassword(auth(), email, password);
      } catch (e) {
        setError(friendly(e));
      }
    },
    signOut: async () => {
      await fbSignOut(auth());
    },
    refreshMe: async () => {
      if (user) await resolveMe(user);
    },
  };

  return <AuthContext.Provider value={shape}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthShape {
  const ctx = useContext(AuthContext);
  if (ctx === null) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
