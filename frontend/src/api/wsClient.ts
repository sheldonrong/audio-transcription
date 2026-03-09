import type { ServerEvent, StartMessage } from "../types/events";

type WsCallbacks = {
  onOpen?: () => void;
  onEvent?: (event: ServerEvent) => void;
  onClose?: (ev: CloseEvent) => void;
  onError?: (ev: Event) => void;
};

const DEFAULT_CHUNK_SIZE = 256 * 1024;
const MAX_BUFFERED_BYTES = 2 * 1024 * 1024;

export class TranscriptionWsClient {
  private socket: WebSocket;

  constructor(url: string, callbacks: WsCallbacks) {
    this.socket = new WebSocket(url);
    this.socket.onopen = () => callbacks.onOpen?.();
    this.socket.onmessage = (event) => {
      const parsed = JSON.parse(event.data) as ServerEvent;
      callbacks.onEvent?.(parsed);
    };
    this.socket.onerror = (event) => callbacks.onError?.(event);
    this.socket.onclose = (event) => callbacks.onClose?.(event);
  }

  sendStart(start: StartMessage): void {
    this.socket.send(JSON.stringify(start));
  }

  async sendAudio(file: Blob, chunkSize = DEFAULT_CHUNK_SIZE): Promise<void> {
    for (let offset = 0; offset < file.size; offset += chunkSize) {
      const chunk = await file.slice(offset, offset + chunkSize).arrayBuffer();
      this.socket.send(chunk);
      await this.waitForDrain();
    }
  }

  endUpload(): void {
    this.socket.send("__end__");
  }

  close(code?: number, reason?: string): void {
    if (code !== undefined) {
      this.socket.close(code, reason);
      return;
    }
    this.socket.close();
  }

  private async waitForDrain(): Promise<void> {
    while (this.socket.bufferedAmount > MAX_BUFFERED_BYTES) {
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
}
