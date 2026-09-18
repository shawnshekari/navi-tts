#pragma once

// Persistent voice store (DESIGN 3.2): one directory per voice under a
// directory the service owns, reloaded at start. The id is the sanitised
// registered name, so `voice_1`, `voice_2` and the Skyrim voice types share
// one map and a re-registration of the same name is idempotent. Thread-safe;
// a request holds a shared_ptr to the voice it uses, so a replace or delete
// during synthesis cannot pull the embedding out from under it.
//
//   <dir>/<id>/voice.json       name, ref_text, model, sample sha256, dim, created
//   <dir>/<id>/embedding.f32    [dim] little-endian float32
//   <dir>/<id>/sample.wav       the reference audio as uploaded (header repaired)

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace navi::voices {

struct Voice {
    std::string id;
    std::string name;
    std::string ref_text;          // kept for ICL cloning (DESIGN 9, OPEN); unused today
    std::string model;             // model id the embedding was computed with
    std::string sample_sha256;     // of the stored sample bytes
    std::string created;           // ISO 8601 UTC
    double sample_seconds = 0;
    std::vector<float> embedding;
};

using VoicePtr = std::shared_ptr<const Voice>;

class Store {
public:
    // Creates `dir` if needed and loads every voice in it. Voices whose
    // embedding is not `dim` floats are skipped with a warning on stderr
    // (dim <= 0: accept any, for the offline CLI).
    static Store open(const std::string & dir, int dim);

    const std::string & dir() const { return dir_; }
    std::size_t size() const;
    std::vector<std::string> ids() const;               // sorted
    std::vector<VoicePtr> all() const;                  // sorted by id
    VoicePtr find(const std::string & id) const;        // nullptr if absent

    // Adds or replaces `v` (its `id` must come from id_for) and writes it to
    // disk before returning. Throws navi::Error on I/O failure; the map is
    // untouched then.
    VoicePtr put(Voice v, std::span<const std::uint8_t> sample_wav);
    bool remove(const std::string & id);                // false if absent

    static std::string id_for(const std::string & name);
    static std::string sha256_hex(std::span<const std::uint8_t> bytes);

private:
    std::string dir_;
    int dim_ = 0;
    std::unique_ptr<std::mutex> m_ = std::make_unique<std::mutex>();   // movable
    std::map<std::string, VoicePtr> voices_;
};

} // namespace navi::voices
