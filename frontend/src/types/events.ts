export type AcceptedEvent = {
  type: "accepted";
  job_id: string;
  filename: string;
};

export type ProgressEvent = {
  type: "progress";
  job_id: string;
  percent: number | null;
  processed_seconds: number;
  total_estimated_seconds: number | null;
};

export type SegmentEvent = {
  type: "segment";
  job_id: string;
  index: number;
  start: number;
  end: number;
  text: string;
  accumulated_text: string;
};

export type CompleteEvent = {
  type: "complete";
  job_id: string;
  text: string;
  segments_count: number;
  duration_seconds: number | null;
};

export type ErrorEvent = {
  type: "error";
  job_id: string;
  message: string;
};

export type ServerEvent =
  | AcceptedEvent
  | ProgressEvent
  | SegmentEvent
  | CompleteEvent
  | ErrorEvent;

export type StartMessage = {
  type: "start";
  filename: string;
  audio_path?: string;
  language: string;
  model: string;
};

export type VideoConversionProgressEvent = {
  type: "progress";
  job_id: string;
  percent: number | null;
  processed_seconds: number;
  total_estimated_seconds: number | null;
};

export type VideoConversionCompleteEvent = {
  type: "complete";
  job_id: string;
  filename: string;
  path: string;
  url: string;
  duration_seconds: number | null;
};

export type VideoConversionServerEvent =
  | AcceptedEvent
  | VideoConversionProgressEvent
  | VideoConversionCompleteEvent
  | ErrorEvent;

export type VideoConversionStartMessage = {
  type: "start";
  video_path: string;
  target_format: "m4a";
};
