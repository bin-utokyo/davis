import handler from "vinext/server/app-router-entry";

interface Env {
  ASSETS: Fetcher;
}

interface ExecutionContext {
  waitUntil(promise: Promise<unknown>): void;
  passThroughOnException(): void;
}

const worker = {
  fetch(request: Request, env: Env, context: ExecutionContext): Promise<Response> {
    return handler.fetch(request, env, context);
  },
};

export default worker;
