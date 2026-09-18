#pragma once

// HTTP front (DESIGN 3): the OpenAI dialect and the XTTS dialect SkyrimNet
// speaks, on 127.0.0.1:8080 (and optionally a second port, so SkyrimNet's
// XTTS.yaml can keep pointing at :8020). One serving thread owns the GPU and
// runs one request at a time; cpp-httplib's accept threads only parse and
// wait. Voices come from the persistent store (runtime/voices); both dialects
// resolve to the same map and the same synth path.

#include "model/qwen3tts/graph.h"
#include "runtime/voices/store.h"

#include <string>

namespace navi::server {

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    int xtts_port = 0;           // a second listener with the same routes; 0 = none
    std::string xtts_fallback_male = "default";     // XTTS speaker names with no voice: by the male/female prefix
    std::string xtts_fallback_female = "default";
    std::string model_id;        // GET /v1/models, the voices map key; default: model file basename
    std::string default_voice;   // what `voice: "default"` (or no voice) means; default: voice_1, else the first id
    bool verbose = false;        // per-request timing on stderr
    int max_input_chars = 4096;
    int max_audio_tokens = 600;  // default per-request frame budget (DESIGN 2)
};

// Blocks until the server stops (SIGINT/SIGTERM).
int run(qwen3tts::Graph & graph, voices::Store & store, const Options & opt);

} // namespace navi::server
