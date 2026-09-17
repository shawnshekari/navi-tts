// Self-heal state machine unit test (docs/talker_fusion_handoff.md,
// "KNOWN ISSUE" / self-healing).
//
// Pure logic, no GPU: verifies the HipHeal policy that governs WHEN a
// fused HIP component (talker / cp / fused-frame) latched off by a
// GPU-contention barrier timeout is re-attempted, so the fused path
// recovers automatically once the GPU quiets instead of staying pinned to
// the GGML path until a manual restart.
//
// The GPU-level behavior (a probe actually recovering once the GPU goes
// quiet) is exercised manually by saturating then idling the GPU; this
// test pins down the policy that drives that recovery.

#include "tts_transformer.h"

#include <cstdio>

using namespace qwen3_tts;

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__); ++failures; } } while (0)

int main() {
    // 1. Healthy component: always attempted, never probing.
    {
        HipHeal h;
        CHECK(h.begin_request(true) == true, "healthy: attempt");
        CHECK(h.probing == false, "healthy: not probing");
        CHECK(h.latched == false, "healthy: not latched");
    }
    // 2. Not-ready (init failed): never attempted, never probes.
    {
        HipHeal h;
        CHECK(h.begin_request(false) == false, "not-ready: no attempt");
        h.note_failure();
        CHECK(h.begin_request(false) == false, "not-ready: still no attempt");
        CHECK(h.begin_request(false) == false, "not-ready: forever no attempt");
    }
    // 3. First latch: backoff grows to 2, since reset.
    {
        HipHeal h;
        h.begin_request(true);
        h.note_failure();
        CHECK(h.latched == true, "latch: latched");
        CHECK(h.backoff == 2, "latch: backoff 2");
        CHECK(h.since == 0, "latch: since 0");
    }
    // 4. Probe cadence: with backoff 2, probe on the 2nd request after latch.
    {
        HipHeal h;
        h.note_failure();
        CHECK(h.begin_request(true) == false, "cadence: req1 no probe (since 1<2)");
        CHECK(h.probing == false, "cadence: req1 not probing");
        CHECK(h.begin_request(true) == true, "cadence: req2 probe (since 2>=2)");
        CHECK(h.probing == true, "cadence: req2 probing");
        CHECK(h.latched == false, "cadence: probe clears latch");
    }
    // 5. Probe success heals: backoff resets to 1, latched false.
    {
        HipHeal h;
        h.note_failure();
        h.begin_request(true);
        h.begin_request(true);
        CHECK(h.probing == true, "heal: probing");
        h.note_success();
        CHECK(h.backoff == 1, "heal: backoff reset 1");
        CHECK(h.latched == false, "heal: not latched");
        CHECK(h.probing == false, "heal: probing cleared");
        CHECK(h.begin_request(true) == true, "heal: healthy after heal");
        CHECK(h.probing == false, "heal: not probing after heal");
    }
    // 6. Probe failure grows backoff exponentially.
    {
        HipHeal h;
        h.note_failure();  // backoff 2
        int expected = 2;
        for (int i = 0; i < 5; ++i) {
            int guard = 0;
            while (!h.probing && guard++ < 1000) h.begin_request(true);
            CHECK(h.probing == true, "grow: probing");
            h.note_failure();
            expected *= 2;
            if (expected > HipHeal::kMaxBackoff) expected = HipHeal::kMaxBackoff;
            CHECK(h.backoff == expected, "grow: backoff doubles");
        }
    }
    // 7. Backoff caps at kMaxBackoff.
    {
        HipHeal h;
        h.note_failure();  // start latched so the probe loop can engage
        for (int i = 0; i < 20; ++i) {
            int guard = 0;
            while (!h.probing && guard++ < 1000) h.begin_request(true);
            CHECK(h.probing == true, "cap: probing");
            h.note_failure();
        }
        CHECK(h.backoff == HipHeal::kMaxBackoff, "cap: backoff capped");
    }
    // 8. Full recovery: latch under load, then a quiet GPU -> the first
    //    probe succeeds and the component is healthy again.
    {
        HipHeal h;
        CHECK(h.begin_request(true) == true, "recover: req1 attempt");
        h.note_failure();
        CHECK(h.latched == true, "recover: latched after req1");
        CHECK(h.begin_request(true) == false, "recover: req2 no probe (still loaded)");
        CHECK(h.begin_request(true) == true, "recover: req3 probe due");
        CHECK(h.probing == true, "recover: req3 probing");
        h.note_success();  // GPU quiet -> probe passes
        CHECK(h.latched == false, "recover: healed");
        CHECK(h.backoff == 1, "recover: backoff reset");
        for (int i = 0; i < 5; ++i) {
            CHECK(h.begin_request(true) == true, "recover: post-heal attempt");
            CHECK(h.probing == false, "recover: post-heal not probing");
            h.note_success();
        }
    }
    // 9. note_success on a non-probing (healthy) request is a no-op.
    {
        HipHeal h;
        h.begin_request(true);
        h.note_success();
        CHECK(h.backoff == 1, "noop: backoff stays 1");
        CHECK(h.latched == false, "noop: not latched");
    }

    if (failures == 0) {
        fprintf(stderr, "PASS: HipHeal self-heal policy\n");
        return 0;
    }
    fprintf(stderr, "FAIL: %d checks failed\n", failures);
    return 1;
}
