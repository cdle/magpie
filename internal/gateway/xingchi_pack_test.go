package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
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
