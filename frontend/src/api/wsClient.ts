import type {
  ServerEvent,
  StartMessage,
  VideoConversionServerEvent,
  VideoConversionStartMessage,
} from "../types/events";

type WsCallbacks<TEvent> = {
  onOpen?: () => void;
  onEvent?: (event: TEvent) => void;
  onClose?: (ev: CloseEvent) => void;
  onError?: (ev: Event) => void;
};

const DEFAULT_CHUNK_SIZE = 256 * 1024;
const MAX_BUFFERED_BYTES = 2 * 1024 * 1024;

type SendFileOptions = {
  chunkSize?: number;
  onProgress?: (percent: number) => void;
};

export class ChunkedUploadWsClient<TEvent, TStart extends object> {
  private socket: WebSocket;

  constructor(url: string, callbacks: WsCallbacks<TEvent>) {
    this.socket = new WebSocket(url);
    this.socket.onopen = () => callbacks.onOpen?.();
    this.socket.onmessage = (event) => {
      const parsed = JSON.parse(event.data) as TEvent;
      callbacks.onEvent?.(parsed);
    };
    this.socket.onerror = (event) => callbacks.onError?.(event);
    this.socket.onclose = (event) => callbacks.onClose?.(event);
  }

  sendStart(start: TStart): void {
    this.socket.send(JSON.stringify(start));
  }

  async sendFile(file: Blob, options: SendFileOptions = {}): Promise<void> {
    const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
    const totalBytes = file.size;

    for (let offset = 0; offset < file.size; offset += chunkSize) {
      const chunk = await file.slice(offset, offset + chunkSize).arrayBuffer();
      this.socket.send(chunk);
      options.onProgress?.((Math.min(offset + chunk.byteLength, totalBytes) / totalBytes) * 100);
      await this.waitForDrain();
    }

    if (totalBytes === 0) {
      options.onProgress?.(100);
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

export class TranscriptionWsClient extends ChunkedUploadWsClient<ServerEvent, StartMessage> {}

export class VideoConversionWsClient extends ChunkedUploadWsClient<
  VideoConversionServerEvent,
  VideoConversionStartMessage
> {}
