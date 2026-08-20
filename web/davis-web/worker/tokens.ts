const encoder = new TextEncoder();
const decoder = new TextDecoder();

export type SessionToken = {
  kind: "session";
  version: 1;
  revision: string;
  issued_at: number;
  expires_at: number;
  nonce: string;
};

export type DownloadToken = {
  kind: "download";
  version: 1;
  revision: string;
  expires_at: number;
  file_id: string;
  path: string;
  oid: string;
  size: number;
};

export async function signToken(
  payload: SessionToken | DownloadToken,
  secret: string,
  purpose: "session" | "download",
): Promise<string> {
  const body = encodeBase64Url(encoder.encode(JSON.stringify(payload)));
  const signature = await sign(`${purpose}.${body}`, secret);
  return `${body}.${encodeBase64Url(signature)}`;
}

export async function verifyToken<T extends SessionToken | DownloadToken>(
  token: string,
  secret: string,
  purpose: "session" | "download",
): Promise<T | null> {
  const parts = token.split(".");
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  try {
    const body = decodeCanonicalBase64Url(parts[0]);
    const signature = decodeCanonicalBase64Url(parts[1]);
    if (!body || !signature) return null;
    const key = await importHmacKey(secret);
    const valid = await crypto.subtle.verify(
      "HMAC",
      key,
      signature,
      encoder.encode(`${purpose}.${parts[0]}`),
    );
    if (!valid) return null;
    return JSON.parse(decoder.decode(body)) as T;
  } catch {
    return null;
  }
}

export async function codesMatch(candidate: string, expected: string): Promise<boolean> {
  const [candidateHash, expectedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(candidate)),
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
  ]);
  const candidateBytes = new Uint8Array(candidateHash);
  const expectedBytes = new Uint8Array(expectedHash);
  let difference = 0;
  for (let index = 0; index < candidateBytes.length; index += 1) {
    difference |= candidateBytes[index] ^ expectedBytes[index];
  }
  return difference === 0;
}

export function randomNonce(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return encodeBase64Url(bytes);
}

async function sign(value: string, secret: string): Promise<Uint8Array> {
  const key = await importHmacKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(value));
  return new Uint8Array(signature);
}

function importHmacKey(secret: string): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

function encodeBase64Url(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/u, "");
}

function decodeBase64Url(value: string): ArrayBuffer {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0)).buffer;
}

function decodeCanonicalBase64Url(value: string): ArrayBuffer | null {
  const decoded = decodeBase64Url(value);
  return encodeBase64Url(new Uint8Array(decoded)) === value ? decoded : null;
}
