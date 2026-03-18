export {};

declare global {
  interface Window {
    APP_CONFIG?: {
      API_BASE_URL?: string;
      WS_URL?: string;
      V2A_WS_URL?: string;
    };
  }
}
