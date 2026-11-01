package gateway

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// codexSentinelOn is a turn Codex, signed in to ChatGPT, sends on the
// placeholder model an older magpie left in its config.
func codexSentinelOn(s *Server, session, input string) *httptest.ResponseRecorder {
	return codexSubTurn(s, codexSentinel, session, "", input)
}

// codexSentinelAsk is a turn on the sentinel over the gateway's own
// Responses endpoint, where Codex with magpie as its provider sends it.
func codexSentinelAsk(t *testing.T, s *Server, session, input string) (int, string) {
	t.Helper()
	body := `{"model":"` + codexSentinel + `","stream":true,"input":[` + input + `]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	req.Header.Set("Authorization", "Bearer chatgpt-token")
	req.Header.Set("User-Agent", "codex_cli_rs/0.159.0 (Mac OS 26.2.0; arm64)")
	req.Header.Set("session_id", session)
	s.Handler().ServeHTTP(rec, req)
	return rec.Code, rec.Body.String()
}

// an older magpie wrote the placeholder into Codex's config as its model,
// and Codex sends it on every turn of the conversation it started then;
// the ChatGPT backend refuses it outright. The turn goes on the model the
// conversation is on, as its own turns last went, and the relay is asked
// for it — not ChatGPT, which refuses the placeholder without naming a
// model the user can pick.
func TestCodexSentinelGoesOnTheConversationsModel(t *testing.T) {
	up, relay, s := codexFollowSetup(t, false)
	was := CodexSideModel
	CodexSideModel = func() string { return "" }
	t.Cleanup(func() { CodexSideModel = was })

	rec := codexSentinelOn(s, "lead-1", leadAsk)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "on it") {
		t.Fatalf("sentinel turn: %d %s", rec.Code, rec.Body.String())
	}
	if got := up.asked(); len(got) != 0 {
		t.Fatalf("ChatGPT was asked for %v", got)
	}
	reqs := relay.requests()
	if len(reqs) != 2 {
		t.Fatalf("the relay got %d requests: %v", len(reqs), reqs)
	}
	if m := modelOf([]byte(reqs[len(reqs)-1])); m != "gpt-6-astra" {
		t.Fatalf("the relay was last asked for %q", m)
	}
	if r := lastRoute(s); r.Model != "third/gpt-6-astra" {
		t.Fatalf("trace: model %q", r.Model)
	}
}

// A conversation on the placeholder with no turn of its own to follow,
// and a model named in Codex's config magpie can serve, goes on that one.
func TestCodexSentinelGoesOnTheModelsConfigNames(t *testing.T) {
	up := codexSubagentSetup(t)
	relay := relayOn(t, "third")
	was := CodexSideModel
	CodexSideModel = func() string { return "third/gpt-6-astra" }
	t.Cleanup(func() { CodexSideModel = was })
	s := New()

	rec := codexSentinelOn(s, "lead-1", leadAsk)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "on it") {
		t.Fatalf("sentinel turn: %d %s", rec.Code, rec.Body.String())
	}
	if got := up.asked(); len(got) != 0 {
		t.Fatalf("ChatGPT was asked for %v", got)
	}
	reqs := relay.requests()
	if m := modelOf([]byte(reqs[len(reqs)-1])); len(reqs) != 1 || m != "gpt-6-astra" {
		t.Fatalf("the relay got %d requests, the last on %q", len(reqs), m)
	}
}

// A conversation on the placeholder with nothing to fall back on is
// answered, rather than relayed to ChatGPT, which refuses the placeholder
// without naming a model the user can pick: the note says the line an
// older magpie left in the config, and the two ways to be rid of it.
func TestCodexSentinelWithNothingToFallBackOnIsAnswered(t *testing.T) {
	up, relay, s := codexFollowSetup(t, false)
	was := CodexSideModel
	CodexSideModel = func() string { return "" }
	t.Cleanup(func() { CodexSideModel = was })

	rec := codexSentinelOn(s, "other-1", leadAsk)
	if rec.Code != 400 {
		t.Fatalf("sentinel turn: %d %s", rec.Code, rec.Body.String())
	}
	// the note is answered as JSON, so its quotes are escaped there
	if note := codexSentinelNote(); !strings.Contains(rec.Body.String(), note) &&
		!strings.Contains(rec.Body.String(), strings.ReplaceAll(note, `"`, `\"`)) {
		t.Fatalf("answered with %q, not %q", rec.Body.String(), note)
	}
	if got := up.asked(); len(got) != 0 {
		t.Fatalf("ChatGPT was asked for %v", got)
	}
	if reqs := relay.requests(); len(reqs) != 1 {
		t.Fatalf("the relay got %d requests: %v", len(reqs), reqs)
	}
}

// The placeholder is one an older magpie left for Codex alone, and a Chat
// request naming it is answered by the route, which resolves no provider
// for a bare model id, not by the note.
func TestCodexSentinelLeftToChatToAnswer(t *testing.T) {
	fresh(t)
	code, body := post(t, "/v1/chat/completions", `{"model":"`+codexSentinel+
		`","stream":true,"messages":[{"role":"user","content":"hi"}]}`)
	if code != 404 {
		t.Fatalf("chat on the sentinel: %d %s", code, body)
	}
	if note := codexSentinelNote(); strings.Contains(body, note) {
		t.Fatalf("chat on the sentinel answered with the note: %s", body)
	}
}

// A turn on the placeholder Codex sends magpie as its provider, over
// /v1/responses, goes on the model the conversation is on too: the
// gateway's own route, not only the backend's.
func TestCodexSentinelOverTheGatewaysResponses(t *testing.T) {
	up := codexSubagentSetup(t)
	relay := relayOn(t, "third")
	was := CodexSideModel
	CodexSideModel = func() string { return "" }
	t.Cleanup(func() { CodexSideModel = was })
	s := New()
	if rec := codexSubTurn(s, "third/gpt-6-astra", "lead-1", "", leadAsk); rec.Code != 200 {
		t.Fatalf("lead: %d %s", rec.Code, rec.Body.String())
	}

	code, out := codexSentinelAsk(t, s, "lead-1", leadAsk)
	if code != 200 || !strings.Contains(out, "on it") {
		t.Fatalf("sentinel over /v1/responses: %d %s", code, out)
	}
	if got := up.asked(); len(got) != 0 {
		t.Fatalf("ChatGPT was asked for %v", got)
	}
	reqs := relay.requests()
	if m := modelOf([]byte(reqs[len(reqs)-1])); len(reqs) != 2 || m != "gpt-6-astra" {
		t.Fatalf("the relay got %d requests, the last on %q", len(reqs), m)
	}
}
