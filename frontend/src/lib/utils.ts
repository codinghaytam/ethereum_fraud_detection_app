import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function getOrCreateUserId(storageKey = "eg_user_id"): string {
  try {
    const existing = localStorage.getItem(storageKey);
    if (existing) return existing;
  } catch {}
  const id =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `anon_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;
  try {
    localStorage.setItem(storageKey, id);
  } catch {}
  return id;
}
