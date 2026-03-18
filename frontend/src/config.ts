function normalizeUrl(value: string | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function getAppConfig() {
  return window.APP_CONFIG ?? {};
}

function toApiBaseFromWs(wsUrl: string): string {
  try {
    const parsed = new URL(wsUrl);
    const protocol = parsed.protocol === "wss:" ? "https:" : "http:";
    return `${protocol}//${parsed.host}`;
  } catch {
    return "http://localhost:8000";
  }
}

function buildWsUrl(configuredUrl: string | null, expectedPath: string): string {
  if (!configuredUrl) {
    return `ws://localhost:8000${expectedPath}`;
  }

  if (new RegExp(`${expectedPath}/?$`).test(configuredUrl)) {
    return configuredUrl;
  }

  try {
    const parsed = new URL(configuredUrl);
    const pathname = parsed.pathname.replace(/\/+$/, "");
    if (!pathname || pathname === "/") {
      return `${parsed.protocol}//${parsed.host}${expectedPath}`;
    }
    return `${parsed.protocol}//${parsed.host}${pathname}`;
  } catch {
    return configuredUrl;
  }
}

export function getTranscriptionWsUrl(): string {
  return buildWsUrl(normalizeUrl(getAppConfig().WS_URL), "/ws/transcribe");
}

export function getV2AWsUrl(): string {
  const appConfig = getAppConfig();
  return buildWsUrl(normalizeUrl(appConfig.V2A_WS_URL ?? appConfig.WS_URL), "/ws/v2a");
}

export function getApiBaseUrl(): string {
  const configured = normalizeUrl(getAppConfig().API_BASE_URL);
  if (configured) {
    return configured.replace(/\/+$/, "");
  }
  return toApiBaseFromWs(getTranscriptionWsUrl());
}
