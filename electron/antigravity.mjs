import { randomUUID } from "node:crypto";
import {
	AgyRequestSessionStore, ANTIGRAVITY_ENDPOINT, ANTIGRAVITY_REDIRECT_URI,
	applyAgyNativeGenerationConfig, authorizeAntigravity, buildAgyAgentRequestMetadata,
	buildAntigravityHarnessUserAgent, ensureProjectContext, exchangeAntigravity,
	fetchWithAgyCliTransport, getPublicModelDefinitions, isGeminiArgonModel,
	isRetiredAgyModel, orderAgyRequestPayloadInPlace, parseRefreshParts,
	refreshAntigravityToken, resolveAntigravityModel, toGeminiSchema,
} from "@cortexkit/antigravity-auth-core";
import {
	AssistantMessageEventStream, createProvider, getCurrentSystemPrompt, getCurrentTools,
} from "@earendil-works/pi-ai";
import { convertMessages } from "@earendil-works/pi-ai/api/google-shared";

export const ANTIGRAVITY_ID = "google-antigravity";
export const ANTIGRAVITY_PROJECT_HEADER = "x-lh-antigravity-project";
const sessions = new AgyRequestSessionStore("");
const validProject = (value) => typeof value === "string" && /^[a-zA-Z0-9._:-]{1,256}$/.test(value);

/** Core's CLI transport is required by Antigravity; keep it behind the app's dispatch gate. */
export function antigravityFetch(input, init) {
	return fetchWithAgyCliTransport(String(input), init, { signal: init?.signal });
}

async function login(interaction, oauth) {
	interaction.signal.throwIfAborted();
	const authorization = await oauth.authorize();
	const state = new URL(authorization.url).searchParams.get("state");
	interaction.notify({ type: "auth_url", url: authorization.url });
	const answer = (await interaction.prompt({ type: "manual_code", message: "Paste the Antigravity OAuth callback URL or authorization code:", placeholder: "After Google redirects, copy the URL even if the page cannot load." })).trim();
	interaction.signal.throwIfAborted();
	let code = answer;
	if (/^[a-z][a-z0-9+.-]*:/i.test(answer)) {
		const callback = new URL(answer), expected = new URL(ANTIGRAVITY_REDIRECT_URI);
		if (callback.origin !== expected.origin || callback.pathname !== expected.pathname || callback.searchParams.get("state") !== state || callback.searchParams.has("error")) throw new Error("Invalid Antigravity callback.");
		code = callback.searchParams.get("code") ?? "";
	}
	if (!code || code.length > 8192 || /\s/.test(code)) throw new Error("Invalid Antigravity authorization code.");
	const result = await oauth.exchange(code, state, { signal: interaction.signal });
	interaction.signal.throwIfAborted();
	if (result.type !== "success") throw new Error("Antigravity sign-in failed.");
	const project = await oauth.project({ ...result, type: "oauth" });
	interaction.signal.throwIfAborted();
	if (!validProject(project.effectiveProjectId)) throw new Error("Antigravity project unavailable.");
	return { type: "oauth", access: result.access, refresh: result.refresh, expires: result.expires, projectId: project.effectiveProjectId };
}

async function refresh(credential, signal, oauth) {
	signal.throwIfAborted();
	const parts = parseRefreshParts(credential.refresh);
	const next = await oauth.refresh(parts.refreshToken);
	// Persist rotation even if the owner pauses during refresh; admission is checked by the owner.
	const suffix = credential.refresh.includes("|") ? credential.refresh.slice(credential.refresh.indexOf("|")) : "";
	return { ...credential, ...next, refresh: `${next.refresh}${suffix}` };
}

/**
 * @param {Partial<{authorize: typeof authorizeAntigravity, exchange: typeof exchangeAntigravity, project: typeof ensureProjectContext, refresh: typeof refreshAntigravityToken}>} [overrides]
 * @returns {import("@earendil-works/pi-ai").Provider<"google-generative-ai">}
 */
export function antigravityProvider(overrides) {
	const oauth = { authorize: authorizeAntigravity, exchange: exchangeAntigravity, project: ensureProjectContext, refresh: refreshAntigravityToken, ...overrides };
	return createProvider({
		id: ANTIGRAVITY_ID, name: "Google Antigravity", baseUrl: ANTIGRAVITY_ENDPOINT,
		auth: { oauth: {
			name: "Google Antigravity", isSubscription: true,
			login: (interaction) => login(interaction, oauth),
			refresh: (credential, signal) => refresh(credential, signal, oauth),
			toAuth: async (credential) => {
				if (!validProject(credential.projectId)) throw new Error("Antigravity project unavailable.");
				return { apiKey: credential.access, baseUrl: ANTIGRAVITY_ENDPOINT, headers: { [ANTIGRAVITY_PROJECT_HEADER]: credential.projectId } };
			},
		} },
		models: Object.values(getPublicModelDefinitions()).filter((model) => !model.modalities.output.includes("image") && !isRetiredAgyModel(model.id)).map((model) => ({
			id: model.id, name: model.name, api: "google-generative-ai", provider: ANTIGRAVITY_ID, baseUrl: ANTIGRAVITY_ENDPOINT,
			reasoning: model.reasoning, input: ["text", "image"], contextWindow: model.limit.context, maxTokens: model.limit.output,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			...(model.reasoning ? { thinkingLevelMap: { off: null, minimal: null, low: model.id.includes("gpt-oss") ? null : "low", medium: "medium", high: model.id.includes("gpt-oss") ? null : "high", xhigh: null, max: null } } : {}),
		})),
		api: { stream, streamSimple: stream },
	});
}

function buildRequest(model, context, options) {
	const tier = options?.reasoning === "minimal" ? "low" : ["xhigh", "max"].includes(options?.reasoning) ? "high" : options?.reasoning;
	const resolved = resolveAntigravityModel(model.reasoning && tier ? `${model.id.replace(/-(low|medium|high)$/, "")}-${tier}` : model.id);
	if (isRetiredAgyModel(model.id) || isRetiredAgyModel(resolved.actualModel)) throw new Error("Antigravity model unavailable.");
	const ownCalls = new Set();
	const wireContext = { ...context, messages: context.messages.map((message) => {
		if (message.role !== "assistant" || message.provider !== model.provider || message.model !== model.id) return message;
		for (const block of message.content) if (block.type === "toolCall") ownCalls.add(block.id.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64));
		return { ...message, model: resolved.actualModel };
	}) };
	const contents = convertMessages({ ...model, id: resolved.actualModel }, wireContext).map((content) => ({
		...content,
		// Antigravity replays its own tool responses as model turns, like the Pi extension.
		...(content.parts?.length && content.parts.every((part) => part.functionResponse && ownCalls.has(part.functionResponse.id)) ? { role: "model" } : {}),
	}));
	const request = { contents };
	const systemPrompt = getCurrentSystemPrompt(context.messages), tools = getCurrentTools(context.messages);
	if (systemPrompt) request.systemInstruction = { parts: [{ text: systemPrompt }] };
	if (tools.length) {
		request.tools = [{ functionDeclarations: tools.map((tool) => ({ name: tool.name, description: tool.description, parameters: toGeminiSchema(tool.parameters, { moveNumericConstraintsToDescription: resolved.actualModel.startsWith("gpt-oss-") }) })) }];
		if (!isGeminiArgonModel(resolved.actualModel)) request.toolConfig = { functionCallingConfig: { mode: options?.toolChoice === "none" ? "NONE" : "VALIDATED" } };
	}
	const generationConfig = { maxOutputTokens: options?.maxTokens ?? model.maxTokens };
	if (resolved.thinkingLevel) generationConfig.thinkingConfig = { includeThoughts: true, thinkingLevel: resolved.thinkingLevel };
	else if (typeof resolved.thinkingBudget === "number") generationConfig.thinkingConfig = { includeThoughts: true, thinkingBudget: resolved.thinkingBudget };
	applyAgyNativeGenerationConfig(resolved.actualModel, generationConfig);
	request.generationConfig = generationConfig;
	const key = options?.sessionId ?? `message:${context.messages.find((message) => message.role !== "system")?.timestamp ?? randomUUID()}`;
	const scope = sessions.beginRequest(key);
	const metadata = buildAgyAgentRequestMetadata(scope.session, request, resolved.actualModel, scope.timestamp, { stepCountMode: "cli" });
	request.labels = metadata.labels; request.sessionId = metadata.sessionId;
	orderAgyRequestPayloadInPlace(request);
	return { key, envelope: { project: options?.headers?.[ANTIGRAVITY_PROJECT_HEADER], requestId: metadata.requestId, request, model: resolved.actualModel, userAgent: "antigravity", requestType: "agent" } };
}

async function* chunks(response, finished) {
	if (!response.body) throw new Error("Antigravity stream is empty.");
	const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
	let buffer = "", data = [], deadline;
	try {
		while (true) {
			// ponytail: final usage gets 1s; increase this grace period if Google's accounting arrives later.
			if (finished()) deadline ??= Date.now() + 1000;
			const remaining = deadline === undefined ? undefined : deadline - Date.now();
			if (remaining !== undefined && remaining <= 0) return;
			let timer, next;
			try {
				const pending = reader.read();
				next = remaining !== undefined ? await Promise.race([pending, new Promise((resolve) => { timer = setTimeout(() => resolve({ done: true }), remaining); })]) : await pending;
			} finally { clearTimeout(timer); }
			const { value, done } = next;
			buffer += done ? "\n\n" : value;
			let match;
			while ((match = /\r\n|\r|\n/.exec(buffer)) && (done || match.index + match[0].length < buffer.length || match[0] !== "\r")) {
				const line = buffer.slice(0, match.index); buffer = buffer.slice(match.index + match[0].length);
				if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
				else if (!line && data.length) {
					const text = data.join("\n"); data = [];
					if (text === "[DONE]") return;
					const parsed = JSON.parse(text);
					yield parsed.response ?? parsed;
				}
			}
			if (done) return;
		}
	} finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}

/** @type {import("@earendil-works/pi-ai").StreamFunction<"google-generative-ai">} */
function stream(model, context, options) {
	const events = new AssistantMessageEventStream();
	void (async () => {
		const output = {
			role: "assistant", provider: model.provider, api: model.api, model: model.id, content: [], timestamp: Date.now(), stopReason: "pending",
			usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		};
		let active = -1, pendingSignature, completed = false;
		const close = () => {
			const block = output.content[active];
			if (block) events.push({ type: `${block.type}_end`, contentIndex: active, content: block.text ?? block.thinking, partial: output });
			active = -1;
		};
		try {
			options?.signal?.throwIfAborted();
			if (!options?.apiKey || !validProject(options?.headers?.[ANTIGRAVITY_PROJECT_HEADER])) throw new Error("Antigravity authorization unavailable.");
			const { key, envelope } = buildRequest(model, context, options);
			const replacement = await options?.onPayload?.(envelope, model);
			if (replacement !== undefined && (!replacement || typeof replacement !== "object" || Array.isArray(replacement))) throw new Error("Invalid Antigravity payload.");
			const payload = { ...(replacement ?? envelope), model: envelope.model, project: envelope.project };
			options?.signal?.throwIfAborted();
			const response = await (options?.fetch ?? antigravityFetch)(`${ANTIGRAVITY_ENDPOINT}/v1internal:streamGenerateContent?alt=sse`, {
				method: "POST", signal: options?.signal,
				headers: { Authorization: `Bearer ${options.apiKey}`, "Content-Type": "application/json", "User-Agent": buildAntigravityHarnessUserAgent(), "Accept-Encoding": "gzip" },
				body: JSON.stringify(payload),
			});
			if (!response.ok) { await response.body?.cancel(); throw new Error(`Antigravity request failed: HTTP ${response.status}`); }
			events.push({ type: "start", partial: output });
			for await (const chunk of chunks(response, () => completed)) {
				options?.signal?.throwIfAborted();
				if (chunk.error || chunk.promptFeedback?.blockReason) throw new Error("Antigravity refused the request.");
				if (chunk.usageMetadata) {
					const usage = chunk.usageMetadata;
					output.usage.cacheRead = usage.cachedContentTokenCount ?? output.usage.cacheRead;
					output.usage.input = Math.max(0, (usage.promptTokenCount ?? output.usage.input + output.usage.cacheRead) - output.usage.cacheRead);
					if (usage.candidatesTokenCount !== undefined || usage.thoughtsTokenCount !== undefined) output.usage.output = (usage.candidatesTokenCount ?? 0) + (usage.thoughtsTokenCount ?? 0);
					output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead;
				}
				if (completed) continue;
				const candidate = chunk.candidates?.[0];
				for (const part of candidate?.content?.parts ?? []) {
					if (part.functionCall) {
						close();
						const toolCall = { type: "toolCall", id: part.functionCall.id ?? `call_${randomUUID()}`, name: part.functionCall.name, arguments: part.functionCall.args ?? {}, ...(part.thoughtSignature || pendingSignature ? { thoughtSignature: part.thoughtSignature ?? pendingSignature } : {}) };
						pendingSignature = undefined;
						const index = output.content.push(toolCall) - 1;
						events.push({ type: "toolcall_start", contentIndex: index, partial: output });
						events.push({ type: "toolcall_delta", contentIndex: index, delta: JSON.stringify(toolCall.arguments), partial: output });
						events.push({ type: "toolcall_end", contentIndex: index, toolCall, partial: output });
					} else if (typeof part.text === "string") {
						const type = part.thought ? "thinking" : "text", field = part.thought ? "thinking" : "text";
						if (output.content[active]?.type !== type) {
							close(); active = output.content.push({ type, [field]: "" }) - 1;
							events.push({ type: `${type}_start`, contentIndex: active, partial: output });
						}
						output.content[active][field] += part.text;
						if (part.thoughtSignature) output.content[active][part.thought ? "thinkingSignature" : "textSignature"] = part.thoughtSignature;
						events.push({ type: `${type}_delta`, contentIndex: active, delta: part.text, partial: output });
					} else if (part.thoughtSignature) {
						if (!part.thought && output.content[active]?.type === "text") output.content[active].textSignature = part.thoughtSignature;
						else pendingSignature = part.thoughtSignature;
					}
				}
				if (candidate?.finishReason) {
					if (!["STOP", "MAX_TOKENS"].includes(candidate.finishReason)) throw new Error("Antigravity stopped without a usable response.");
					output.stopReason = candidate.finishReason === "MAX_TOKENS" ? "length" : output.content.some((block) => block.type === "toolCall") ? "toolUse" : "stop";
					completed = true;
				}
			}
			options?.signal?.throwIfAborted();
			if (!completed) throw new Error("Antigravity stream ended without a finish reason.");
			close(); sessions.completeExecution(key);
			events.push({ type: "done", reason: output.stopReason, message: output });
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = /^Antigravity request failed: HTTP \d{3}$/.test(error?.message) ? error.message : "Antigravity request failed. Check the connection and try again.";
			events.push({ type: "error", reason: output.stopReason, error: output });
		} finally { events.end(); }
	})();
	return events;
}
