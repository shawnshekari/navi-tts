#pragma once

// HTTP front (DESIGN 3): the OpenAI dialect on 127.0.0.1:8080. One serving
// thread owns the GPU and runs one request at a time; cpp-httplib's accept
// threads only parse and wait. Voices are in memory until the store (M2).

#include "model/qwen3tts/graph.h"

#include <string>

namespace navi::server {

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string model_id;        // GET /v1/models, the voices map key; default: model file basename
    bool verbose = false;        // per-request timing on stderr
    int max_input_chars = 4096;
    int max_audio_tokens = 600;  // default per-request frame budget (DESIGN 2)
};

// Blocks until the server stops (SIGINT/SIGTERM).
int run(qwen3tts::Graph & graph, const Options & opt);

} // namespace navi::server
