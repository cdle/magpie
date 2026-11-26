package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
	"github.com/yetone/magpie/internal/provider"
)

func xingchiPackFixture(turns int) []byte {
	items := make([]map[string]any, 0, turns*2)
	for i := 1; i <= turns; i++ {
		items = append(items,
			map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": fmt.Sprintf("user-%02d exact text", i)}}},
			map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": fmt.Sprintf("assistant-%02d exact answer", i)}}},
		)
	}
	items = append(items, map[string]any{"type": "message", "role": "user", "content": []any{map[string]any{"type": "input_text", "text": "current-turn"}}})
	body, _ := json.Marshal(map[string]any{"model": "gpt-6.1-sol", "instructions": "unchanged instructions", "input": items, "store": false})
	return body
}

func packedInput(t *testing.T, body []byte) []json.RawMessage {
	t.Helper()
	var q map[string]json.RawMessage
	if err := json.Unmarshal(body, &q); err != nil {
		t.Fatal(err)
	}
	items, ok := inputList(q["input"])
	if !ok {
		t.Fatal("packed body has no input list")
	}
	return items
}

func TestXingchiPackKeepsTenOldTurnsAndCurrentInput(t *testing.T) {
	in := http.Header{}
	in.Set("Thread-Id", "pack-thread")
	body := xingchiPackBody(in, xingchiPackFixture(12))
	if got := gjson.GetBytes(body, "instructions").String(); got != "unchanged instructions" {
		t.Fatalf("instructions = %q", got)
	}
	items := packedInput(t, body)
	if len(items) != 1+10*2+1 {
		t.Fatalf("wire input has %d items; want one pack, ten complete old turns, and current input", len(items))
	}
	var pack struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(items[0], &pack); err != nil || pack.Role != "developer" {
		t.Fatalf("pack item = %s, err %v", items[0], err)
	}
	if !strings.HasPrefix(pack.Content, xingchiPackMarker+"\n") {
		t.Fatalf("package marker missing: %.40q", pack.Content)
	}
	if !strings.Contains(pack.Content, "user-01 exact text") || !strings.Contains(pack.Content, "assistant-02 exact answer") {
		t.Fatalf("older turns are missing from the package: %s", pack.Content)
	}
	parts := make([][]byte, len(items))
	for i := range items {
		parts[i] = items[i]
	}
	joined := string(bytes.Join(parts[1:], []byte("\n")))
	if strings.Contains(joined, "user-01 exact text") || !strings.Contains(joined, "user-12 exact text") || !strings.Contains(joined, "current-turn") {
		t.Fatalf("recent turns or current input are not in the expected positions")
	}
	if got := gjson.GetBytes(body, "input.#").Int(); got != int64(len(items)) {
		t.Fatalf("wire input count = %d", got)
	}
}

func TestXingchiPackDoesNotChangeOtherRoutes(t *testing.T) {
	body := xingchiPackFixture(14)
	if got := xingchiPackBody(http.Header{}, body); !strings.Contains(string(got), xingchiPackMarker) {
		t.Fatal("fallback input hash did not allow packing")
	}
	// The routing scope is enforced by forwardOnce; the helper is deliberately
	// route-agnostic so it can be tested without provider-global setup.
}

func TestXingchiPackAppendsStablePrefix(t *testing.T) {
	in := http.Header{}
	in.Set("Thread-Id", "stable-prefix-thread")
	first := xingchiPackBody(in, xingchiPackFixture(12))
	second := xingchiPackBody(in, xingchiPackFixture(13))
	firstItems, secondItems := packedInput(t, first), packedInput(t, second)
	var a, b struct {
		Content string `json:"content"`
	}
	json.Unmarshal(firstItems[0], &a)
	json.Unmarshal(secondItems[0], &b)
	trailer := strings.LastIndex(a.Content, "\n[magpie-packed end turns=")
	if trailer < 0 || !strings.HasPrefix(b.Content, a.Content[:trailer]) {
		t.Fatal("a later package did not preserve the earlier package body as a stable prefix")
	}
	if !strings.HasPrefix(b.Content, xingchiPackMarker+"\n") || !strings.HasPrefix(a.Content, xingchiPackMarker+"\n") {
		t.Fatal("the sentinel moved or changed")
	}
}

func TestXingchiPackUnpacksAndChecksChecksum(t *testing.T) {
	in := http.Header{}
	in.Set("Thread-Id", "unpack-thread")
	body := xingchiPackBody(in, xingchiPackFixture(12))
	items := packedInput(t, body)
	var packed json.RawMessage
	packed = items[0]
	got := xingchiUnpackInput([]json.RawMessage{packed})
	if len(got) != 4 {
		t.Fatalf("unpacked %d items, want two older user/assistant turns", len(got))
	}
	if !strings.Contains(string(got[0]), "user-01 exact text") || !strings.Contains(string(got[1]), "assistant-01 exact answer") {
		t.Fatalf("round-trip lost original text: %s", got)
	}
	var pack struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	json.Unmarshal(packed, &pack)
	pack.Content = strings.Replace(pack.Content, "user-01 exact text", "tampered", 1)
	broken, _ := json.Marshal(pack)
	got = xingchiUnpackInput([]json.RawMessage{broken})
	if len(got) != 1 || !strings.Contains(string(got[0]), xingchiPackMarker) {
		t.Fatal("checksum failure should leave the item as ordinary text")
	}
}

func TestXingchiPackOmitsOldImagesAndReasoning(t *testing.T) {
	items := []any{}
	for i := 1; i <= 12; i++ {
		content := []any{map[string]any{"type": "input_text", "text": fmt.Sprintf("turn-%02d", i)}}
		if i == 1 {
			content = append(content, map[string]any{"type": "input_image", "image_url": "data:image/png;base64,PRIVATE-IMAGE-DATA"})
		}
		items = append(items,
			map[string]any{"type": "message", "role": "user", "content": content},
			map[string]any{"type": "reasoning", "summary": []any{map[string]any{"type": "summary_text", "text": "private reasoning"}}},
			map[string]any{"type": "message", "role": "assistant", "content": []any{map[string]any{"type": "output_text", "text": "answer"}}},
		)
	}
	items = append(items, map[string]any{"type": "message", "role": "user", "content": "current"})
	body, _ := json.Marshal(map[string]any{"input": items})
	in := http.Header{}
	in.Set("Thread-Id", "media-thread")
	packed := xingchiPackBody(in, body)
	var pack struct {
		Content string `json:"content"`
	}
	json.Unmarshal(packedInput(t, packed)[0], &pack)
	if strings.Contains(pack.Content, "PRIVATE-IMAGE-DATA") || strings.Contains(pack.Content, "private reasoning") {
		t.Fatal("old image bytes or reasoning leaked into the packed message")
	}
	if !strings.Contains(string(packed), "[image omitted: sha256=") {
		t.Fatal("old image has no placeholder")
	}
}

func xingchiAgentFixture(iterations, calls int, user bool) []byte {
	items := []map[string]any{{"type": "message", "role": "developer", "content": "keep instructions"}}
	if user {
		items = append(items, map[string]any{"type": "message", "role": "user", "content": "one initial request"})
	}
	for i := 0; i < iterations; i++ {
		items = append(items, map[string]any{"type": "message", "role": "assistant", "content": fmt.Sprintf("iteration-%03d", i)})
		for j := 0; j < calls; j++ {
			items = append(items, map[string]any{"type": "function_call", "call_id": fmt.Sprintf("call-%d-%d", i, j), "name": "probe", "arguments": fmt.Sprintf(`{"iteration":%d}`, i)})
		}
		for j := 0; j < calls; j++ {
			items = append(items, map[string]any{"type": "function_call_output", "call_id": fmt.Sprintf("call-%d-%d", i, j), "output": fmt.Sprintf("result-%d-%d", i, j)})
		}
	}
	body, _ := json.Marshal(map[string]any{"input": items, "model": "gpt-6.1-sol"})
	return body
}

func xingchiTestAssistantItems(items []json.RawMessage) int {
	n := 0
	for _, raw := range items {
		if gjson.GetBytes(raw, "role").String() == "assistant" || gjson.GetBytes(raw, "type").String() == "function_call" || gjson.GetBytes(raw, "type").String() == "custom_tool_call" {
			n++
		}
	}
	return n
}

func TestXingchiPackAgentHistory(t *testing.T) {
	for _, tc := range []struct {
		name              string
		iterations, calls int
		user              bool
	}{
		{"single-user", 110, 1, true},
		{"parallel-tools", 15, 8, true},
		{"oversized-latest-iteration", 1, 60, true},
		{"assistant-only", 110, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := xingchiAgentFixture(tc.iterations, tc.calls, tc.user)
			packed := xingchiPackBody(http.Header{"Thread-Id": {t.Name()}}, body)
			items := packedInput(t, packed)
			if bytes.Equal(body, packed) || strings.Count(string(packed), xingchiPackMarker) != 1 {
				t.Fatal("agent history must produce exactly one pack")
			}
			if n := xingchiTestAssistantItems(items); n > 50 {
				t.Fatalf("wire assistant items = %d, want <=50", n)
			}
			if string(items[0]) != string(packedInput(t, body)[0]) {
				t.Fatal("leading instructions changed")
			}
			if tc.iterations > 1 && !bytes.Contains(items[len(items)-1], []byte(fmt.Sprintf("%d", tc.iterations-1))) {
				t.Fatal("latest iteration did not stay native")
			}
			calls := map[string]bool{}
			for _, raw := range items {
				switch gjson.GetBytes(raw, "type").String() {
				case "function_call":
					calls[gjson.GetBytes(raw, "call_id").String()] = true
				case "function_call_output":
					if !calls[gjson.GetBytes(raw, "call_id").String()] {
						t.Fatal("native tool result lost its call")
					}
				}
			}
			unpacked := xingchiUnpackInput(items)
			original := packedInput(t, body)
			if len(unpacked) != len(original) {
				t.Fatalf("round trip items = %d, want %d", len(unpacked), len(original))
			}
			for i := range original {
				if !bytes.Equal(unpacked[i], original[i]) {
					t.Fatalf("round trip item %d changed: %s", i, unpacked[i])
				}
			}
			again := xingchiPackBody(http.Header{"Thread-Id": {t.Name()}}, packed)
			if !bytes.Equal(again, packed) || strings.Count(string(again), xingchiPackMarker) != 1 {
				t.Fatal("repacking nested a pack or lost tool history")
			}
		})
	}
}

func TestXingchiPackAssistantBudgetBoundary(t *testing.T) {
	for _, calls := range []int{49, 50} {
		body := xingchiAgentFixture(1, calls, true)
		got := xingchiPackBody(http.Header{"Thread-Id": {t.Name()}}, body)
		if calls == 49 && !bytes.Equal(got, body) {
			t.Fatal("50 assistant items should remain native")
		}
		if calls == 50 && (bytes.Equal(got, body) || xingchiTestAssistantItems(packedInput(t, got)) > 50) {
			t.Fatal("51 assistant items must be packed into budget")
		}
	}
}

func TestXingchiPackAgentStablePrefix(t *testing.T) {
	in := http.Header{"Thread-Id": {t.Name()}}
	first := xingchiPackBody(in, xingchiAgentFixture(60, 1, true))
	second := xingchiPackBody(in, xingchiAgentFixture(61, 1, true))
	a := gjson.GetBytes(first, "input.1.content").String()
	b := gjson.GetBytes(second, "input.1.content").String()
	trailer := strings.LastIndex(a, "\n[magpie-packed end turns=")
	if trailer < 0 || !strings.HasPrefix(b, a[:trailer+1]) {
		t.Fatal("agent pack rewrote its earlier prefix")
	}
}

func TestXingchiPackBadAgentChecksumIsOrdinaryText(t *testing.T) {
	body := xingchiPackBody(http.Header{"Thread-Id": {t.Name()}}, xingchiAgentFixture(60, 1, true))
	items := packedInput(t, body)
	var pack map[string]any
	if len(items) < 2 || json.Unmarshal(items[1], &pack) != nil {
		t.Fatal("missing pack")
	}
	text, _ := pack["content"].(string)
	if !strings.HasPrefix(text, xingchiPackMarker+"\n") {
		t.Fatal("missing agent sentinel")
	}
	pack["content"] = strings.Replace(text, "iteration-000", "tampered", 1)
	broken, _ := json.Marshal(pack)
	got := xingchiUnpackInput([]json.RawMessage{broken})
	if len(got) != 1 || !bytes.Equal(got[0], broken) {
		t.Fatal("checksum mismatch must preserve ordinary developer text")
	}
}

func TestXingchiPackToolOnlyIterations(t *testing.T) {
	for _, custom := range []bool{false, true} {
		items := packedInput(t, xingchiAgentFixture(110, 1, true))
		var input []json.RawMessage
		for _, item := range items {
			if gjson.GetBytes(item, "role").String() == "assistant" {
				continue
			}
			if custom {
				item = bytes.ReplaceAll(item, []byte(`"function_call`), []byte(`"custom_tool_call`))
			}
			input = append(input, item)
		}
		body, _ := json.Marshal(map[string]any{"input": input})
		got := packedInput(t, xingchiPackBody(http.Header{}, body))
		if n := xingchiTestAssistantItems(got); n != 11 {
			t.Fatalf("custom=%v: native calls=%d, want 11 recent iterations", custom, n)
		}
		if n := len(xingchiUnpackInput(got)); n != len(input) {
			t.Fatalf("custom=%v: unpacked %d items, want %d", custom, n, len(input))
		}
	}
}

func TestForwardOncePacksOnlyXingchiResponses(t *testing.T) {
	body := xingchiAgentFixture(110, 1, true)
	for _, tc := range []struct {
		id       string
		protocol provider.Protocol
		path     string
		pack     bool
	}{
		{"xingchi", provider.Responses, "/responses", true},
		{"other", provider.Responses, "/responses", false},
		{"xingchi", provider.Chat, "/chat/completions", false},
	} {
		t.Run(tc.id+tc.path, func(t *testing.T) {
			captured := make(chan []byte, 1)
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, _ := io.ReadAll(r.Body)
				captured <- b
				w.Header().Set("Content-Type", "application/json")
				io.WriteString(w, `{}`)
			}))
			defer up.Close()
			p := provider.Provider{ID: tc.id, Responses: up.URL + "/v1", Chat: up.URL + "/v1", Key: "fixture-key"}
			res, err := New().forwardOnce(context.Background(), p, tc.protocol, tc.path, body, http.Header{})
			if err != nil {
				t.Fatal(err)
			}
			res.Body.Close()
			wire := <-captured
			if tc.pack {
				if !bytes.Contains(wire, []byte(xingchiPackMarker)) || xingchiTestAssistantItems(packedInput(t, wire)) > 50 {
					t.Fatal("final upstream boundary did not pack agent input into budget")
				}
			} else if !bytes.Equal(wire, body) {
				t.Fatal("another route changed the original input")
			}
		})
	}
}
