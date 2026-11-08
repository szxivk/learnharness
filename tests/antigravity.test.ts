import assert from "node:assert/strict";
import test from "node:test";
import { createModels, getSupportedThinkingLevels, normalizeContext, type OAuthCredential, type ProviderAuthInteraction } from "@earendil-works/pi-ai";
import { ANTIGRAVITY_ENDPOINT, ANTIGRAVITY_REDIRECT_URI, authorizeAntigravity } from "@cortexkit/antigravity-auth-core";
import { ANTIGRAVITY_PROJECT_HEADER, antigravityProvider } from "../electron/antigravity.mjs";

const credential = (): OAuthCredential => ({ type: "oauth", access: "fixture-access", refresh: "fixture-refresh|fixture-project", expires: Date.now() + 3600000, projectId: "fixture-project" });
const context = normalizeContext({ systemPrompt: "Teach carefully.", tools: [{ name: "lookup", description: "Look up a fact", parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } }], messages: [{ role: "user", content: "Explain this.", timestamp: 1 }] });
function fixture() {
	let exchanges = 0, refreshes = 0, opened = "";
	const provider = antigravityProvider({
		authorize: authorizeAntigravity,
		exchange: async (code, state, options) => { options?.signal?.throwIfAborted(); assert.equal(code, "fixture-code"); assert(state); exchanges++; return { type: "success", access: "fixture-access", refresh: "fixture-refresh|fixture-project", expires: Date.now() + 3600000, projectId: "fixture-project" }; },
		project: async (auth) => ({ auth, effectiveProjectId: "fixture-project" }),
		refresh: async (token) => { assert.equal(token, "fixture-refresh"); refreshes++; return { access: "fixture-new-access", refresh: "fixture-rotated", expires: Date.now() + 3600000 }; },
	});
	const interaction = (respond?: (url: string) => string): ProviderAuthInteraction => ({
		signal: new AbortController().signal,
		notify: (event) => { if (event.type === "auth_url") opened = event.url; },
		prompt: async () => {
			if (respond) return respond(opened);
			const callback = new URL(ANTIGRAVITY_REDIRECT_URI);
			callback.searchParams.set("code", "fixture-code"); callback.searchParams.set("state", new URL(opened).searchParams.get("state")!);
			return callback.href;
		},
	});
	return { provider, interaction, get exchanges() { return exchanges; }, get refreshes() { return refreshes; } };
}

test("Antigravity OAuth validates callback state and origin, supports raw codes, and preserves rotation/project", async () => {
	const f = fixture(), oauth = f.provider.auth.oauth!;
	assert.equal(f.provider.auth.apiKey, undefined);
	assert.equal((await oauth.login(f.interaction())).projectId, "fixture-project");
	assert.equal((await oauth.login(f.interaction(() => "fixture-code"))).type, "oauth");
	for (const answer of ["http://localhost:51121/oauth-callback?code=fixture-code&state=wrong", "https://example.test/?code=fixture-code", "", "invalid code"]) await assert.rejects(oauth.login(f.interaction(() => answer)));
	assert.equal(f.exchanges, 2);
	const next = await oauth.refresh(credential(), new AbortController().signal);
	assert.equal(next.refresh, "fixture-rotated|fixture-project"); assert.equal(next.projectId, "fixture-project");
	assert.equal((await oauth.toAuth(next)).headers?.[ANTIGRAVITY_PROJECT_HEADER], "fixture-project"); assert.equal(f.refreshes, 1);
	await assert.rejects(oauth.toAuth({ ...credential(), projectId: "bad\nproject" }));
	const aborted = new AbortController(); aborted.abort();
	await assert.rejects(oauth.login({ ...f.interaction(), signal: aborted.signal }));
	await assert.rejects(oauth.refresh(credential(), aborted.signal)); assert.equal(f.refreshes, 1);
});

test("Antigravity catalog is separate from Google API models and does not offer unsupported GPT effort", () => {
	const p = antigravityProvider(), models = createModels(); models.setProvider(p);
	assert(p.getModels().length > 0); assert(p.getModels().every((model) => model.provider === "google-antigravity" && model.input.includes("image")));
	assert.deepEqual(getSupportedThinkingLevels(p.getModels().find((model) => model.id.includes("gpt-oss"))!), ["medium"]);
	assert.deepEqual(getSupportedThinkingLevels(p.getModels().find((model) => model.id.includes("gemini"))!), ["low", "medium", "high"]);
});

function response(frames: unknown[], truncate = false) {
	const text = frames.map((frame) => `data: ${JSON.stringify({ response: frame })}\r\n\r\n`).join("") + (truncate ? "data: {broken\n\n" : "");
	const bytes = new TextEncoder().encode(text);
	return new Response(new ReadableStream({ start(controller) { for (let i = 0; i < bytes.length; i += 3) controller.enqueue(bytes.slice(i, i + 3)); controller.close(); } }));
}

test("Antigravity streams split UTF-8/SSE, thoughts, tool calls, signatures and trailing usage with selected identity", async () => {
	const p = antigravityProvider(), model = p.getModels().find((model) => model.id.includes("gemini-3.8"))!;
	let sends = 0;
	const stream = p.streamSimple(model, context, {
		apiKey: "fixture-access", headers: { [ANTIGRAVITY_PROJECT_HEADER]: "fixture-project" }, reasoning: "low",
		onPayload: (raw) => ({ ...raw as object, model: "cannot-change-model", project: "cannot-change-project" }),
		fetch: async (url, init) => {
			sends++; assert.equal(String(url), `${ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse`);
			const headers = new Headers(init?.headers); assert.equal(headers.get("Authorization"), "Bearer fixture-access"); assert(!headers.has(ANTIGRAVITY_PROJECT_HEADER));
			const payload = JSON.parse(init!.body as string);
			assert.equal(payload.model, "gemini-3.8-flash-low"); assert.equal(payload.project, "fixture-project");
			assert.equal(payload.request.systemInstruction.parts[0].text, "Teach carefully.");
			assert.equal(payload.request.tools[0].functionDeclarations[0].parameters.type, "OBJECT");
			assert.equal(payload.request.toolConfig.functionCallingConfig.mode, "VALIDATED");
			return response([
				{ candidates: [{ content: { parts: [{ text: "Think", thought: true, thoughtSignature: "thinking-signature" }] } }] },
				{ candidates: [{ content: { parts: [{ text: "Café", thoughtSignature: "text-signature" }, { functionCall: { name: "lookup", args: { query: "fact" }, id: "call-1" }, thoughtSignature: "tool-signature" }] }, finishReason: "STOP" }], usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 4, thoughtsTokenCount: 2, cachedContentTokenCount: 3 } },
				{ usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5, thoughtsTokenCount: 2, cachedContentTokenCount: 3 } },
			]);
		},
	});
	const events = []; for await (const event of stream) events.push(event.type);
	const result = await stream.result();
	assert.equal(sends, 1); assert.equal(result.stopReason, "toolUse"); assert.equal(result.provider, p.id);
	assert.deepEqual(result.content, [{ type: "thinking", thinking: "Think", thinkingSignature: "thinking-signature" }, { type: "text", text: "Café", textSignature: "text-signature" }, { type: "toolCall", id: "call-1", name: "lookup", arguments: { query: "fact" }, thoughtSignature: "tool-signature" }]);
	assert.equal(result.usage.input, 7); assert.equal(result.usage.output, 7); assert.equal(result.usage.totalTokens, 17); assert.equal(events.at(-1), "done");
});

test("Antigravity replays signed tool calls/results for Gemini, Claude and GPT without losing their IDs", async () => {
	const p = antigravityProvider();
	for (const name of ["gemini-3.8", "claude-sonnet", "gpt-oss"]) {
		const model = p.getModels().find((entry) => entry.id.includes(name))!;
		const history = normalizeContext({ messages: [
			{ role: "user", content: "Look it up.", timestamp: 1 },
			{ role: "assistant", provider: p.id, api: model.api, model: model.id, timestamp: 2, stopReason: "toolUse", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, content: [{ type: "toolCall", id: "call_1", name: "lookup", arguments: { query: "fact" }, thoughtSignature: "c2lnbmF0dXJl" }] },
			{ role: "toolResult", toolCallId: "call_1", toolName: "lookup", content: [{ type: "text", text: "A fact." }], isError: false, timestamp: 3 },
		] });
		const result = await p.streamSimple(model, history, {
			apiKey: "fixture-access", headers: { [ANTIGRAVITY_PROJECT_HEADER]: "fixture-project" },
			fetch: async (_url, init) => {
				const contents = JSON.parse(init!.body as string).request.contents;
				assert.equal(contents[1].parts[0].functionCall.id, "call_1");
				assert.equal(contents[1].parts[0].thoughtSignature, "c2lnbmF0dXJl");
				assert.equal(contents[2].parts[0].functionResponse.id, "call_1"); assert.equal(contents[2].role, "model");
				return response([{ candidates: [{ content: { parts: [{ text: "Done." }] }, finishReason: "STOP" }] }]);
			},
		}).result();
		assert.equal(result.stopReason, "stop", name);
	}
});

test("Antigravity finishes after the trailing-usage grace period when Google leaves the stream open", async () => {
	const p = antigravityProvider(); let cancelled = false;
	const result = await p.streamSimple(p.getModels()[0]!, context, {
		apiKey: "fixture-access", headers: { [ANTIGRAVITY_PROJECT_HEADER]: "fixture-project" },
		fetch: async () => new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(`data: ${JSON.stringify({ response: { candidates: [{ content: { parts: [{ text: "Done." }] }, finishReason: "STOP" }] } })}\n\n`)); }, cancel() { cancelled = true; } })),
	}).result();
	assert.equal(result.stopReason, "stop"); assert(cancelled);
});

test("Antigravity rejects failed, blocked, truncated and cancelled streams without leaking provider details or retrying", async () => {
	const p = antigravityProvider(), model = p.getModels()[0]!;
	for (const make of [() => new Response("fixture-private-provider-detail", { status: 401 }), () => response([{ promptFeedback: { blockReason: "SAFETY" } }]), () => response([{ candidates: [{ content: { parts: [{ text: "partial" }] } }] }]), () => response([], true), () => response([{ candidates: [{ finishReason: "SAFETY" }] }])]) {
		let sends = 0;
		const result = await p.streamSimple(model, context, { apiKey: "fixture-access", headers: { [ANTIGRAVITY_PROJECT_HEADER]: "fixture-project" }, fetch: async () => { sends++; return make(); } }).result();
		assert.equal(result.stopReason, "error"); assert.equal(sends, 1); assert(!JSON.stringify(result).includes("fixture-private-provider-detail"));
	}
	const abort = new AbortController(); abort.abort();
	const cancelled = await p.streamSimple(model, context, { signal: abort.signal, fetch: async () => { throw new Error("must not dispatch"); } }).result();
	assert.equal(cancelled.stopReason, "aborted");
});
