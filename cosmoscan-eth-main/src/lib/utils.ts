import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { AddressPredictionResponse } from "./types"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}


// API function to get address prediction
export async function getAddressPrediction(address: string): Promise<AddressPredictionResponse> {
  const response = await fetch('http://backend:8080/api/analyse?address='+address, {
    method: 'POST',
    
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status, response.statusText}`);
  }

  return response.json();
}