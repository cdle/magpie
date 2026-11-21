package gateway

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

// TestForwardOnceStripsXingchiSession: a Responses request to Xingchi
// goes upstream with none of the session identifiers Xingchi counts
// against its per-session request cap — the session-id and
// x-session-id headers and the body's prompt_cache_key — while its
// thread-id, the body's session_id and the provider's own signing stay
// as they were. The strip runs in forwardOnce after Sign, so what is
// captured here is the request as it goes on the wire.
func TestForwardOnceStripsXingchiSession(t *testing.T) {
	var mu sync.Mutex
	var gotHeader http.Header
	var gotBody []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		gotHeader, gotBody = r.Header.Clone(), b
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{}`)
	}))
	defer up.Close()

	in := http.Header{}
	in.Set("User-Agent", "codex_cli_rs/0.1.0")
	in.Set("originator", "codex_cli_rs")
	in.Set("session-id", "sess-exhausted")
	in.Set("x-session-id", "sess-exhausted-x")
	in.Set("thread-id", "thr-1")

	s := New()
	p := provider.Provider{ID: "xingchi", Responses: up.URL + "/v1", Key: "provider-key"}
	body := []byte(`{"model":"gpt-6.1-sol","input":"hi","prompt_cache_key":"sess-exhausted","session_id":"keep-me"}`)
	res, err := s.forwardOnce(context.Background(), p, provider.Responses, "/responses", body, in)
	if err != nil {
		t.Fatal(err)
	}
	res.Body.Close()

	mu.Lock()
	defer mu.Unlock()
	if v := gotHeader.Get("Session-Id"); v != "" {
		t.Errorf("Session-Id = %q, want none: Xingchi counts it against the session's request cap", v)
	}
	if v := gotHeader.Get("X-Session-Id"); v != "" {
		t.Errorf("X-Session-Id = %q, want none: Xingchi counts it against the session's request cap", v)
	}
	if v := gotHeader.Get("Thread-Id"); v != "thr-1" {
		t.Errorf("Thread-Id = %q, want thr-1: only the session identifiers come off", v)
	}
	if gjson.GetBytes(gotBody, "prompt_cache_key").Exists() {
		t.Errorf("body still carries prompt_cache_key %s: Xingchi counts it against the session's request cap", gotBody)
	}
	if v := gjson.GetBytes(gotBody, "session_id").String(); v != "keep-me" {
		t.Errorf("body session_id = %q, want keep-me: it is not one of the identifiers Xingchi caps", v)
	}
	if !strings.HasPrefix(gotHeader.Get("Authorization"), "Bearer ") {
		t.Errorf("Authorization = %q, want the provider's signing left on", gotHeader.Get("Authorization"))
	}
}

// TestForwardOnceKeepsSessionForOthers: the strip is Xingchi's alone,
// and only for Responses. Another provider's Responses request, and a
// Chat request to Xingchi itself, keep the client's session-id and
// prompt_cache_key exactly as sent.
func TestForwardOnceKeepsSessionForOthers(t *testing.T) {
	var mu sync.Mutex
	var got []struct {
		header http.Header
		body   []byte
	}
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		got = append(got, struct {
			header http.Header
			body   []byte
		}{r.Header.Clone(), b})
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{}`)
	}))
	defer up.Close()

	in := http.Header{}
	in.Set("User-Agent", "codex_cli_rs/0.1.0")
	in.Set("originator", "codex_cli_rs")
	in.Set("session-id", "sess-keep")

	s := New()
	body := []byte(`{"model":"gpt-6.1-sol","input":"hi","prompt_cache_key":"sess-keep"}`)
	for _, tc := range []struct {
		p    provider.Provider
		to   provider.Protocol
		path string
	}{
		{provider.Provider{ID: "other", Responses: up.URL + "/v1", Key: "k"}, provider.Responses, "/responses"},
		{provider.Provider{ID: "xingchi", Chat: up.URL + "/v1", Key: "k"}, provider.Chat, "/chat/completions"},
	} {
		res, err := s.forwardOnce(context.Background(), tc.p, tc.to, tc.path, body, in)
		if err != nil {
			t.Fatal(err)
		}
		res.Body.Close()
	}

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("%d requests reached the upstream, want 2", len(got))
	}
	for i, g := range got {
		if v := g.header.Get("Session-Id"); v != "sess-keep" {
			t.Errorf("request %d: Session-Id = %q, want sess-keep: the strip is Xingchi Responses alone", i, v)
		}
		if v := gjson.GetBytes(g.body, "prompt_cache_key").String(); v != "sess-keep" {
			t.Errorf("request %d: prompt_cache_key = %q, want sess-keep: the strip is Xingchi Responses alone", i, v)
		}
	}
}
