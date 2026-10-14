#pragma once

// Persistent voice store (DESIGN 3.2): one directory per voice under a
// directory the service owns, reloaded at start. The id is the sanitised
// registered name, so `voice_1`, `voice_2` and the Skyrim voice types share
// one map and a re-registration of the same name is idempotent. Thread-safe;
// a request holds a shared_ptr to the voice it uses, so a replace or delete
// during synthesis cannot pull the embedding out from under it.
//
//   <dir>/<id>/voice.json       name, ref_text, model, sample sha256, dim, created, and
//                               the payload hashes below
//   <dir>/<id>/embedding.f32    [dim] little-endian float32
//   <dir>/<id>/ref_codes.i32    [ref_frames][16] little-endian int32, in-context cloning
//                               (absent for a voice without a transcript, or one cloned by
//                               a build that had no codec encoder)
//   <dir>/<id>/sample.wav       the reference audio as uploaded (header repaired)
//
// The manifest is written last and carries `embedding_sha256` / `ref_codes_sha256`, so a
// crash between the payload writes and the manifest leaves a manifest that describes the
// OLD payloads and the mismatch is caught at load: the voice is skipped if its embedding
// does not match, and demoted to embedding-only if its codes do not. That is detection,
// not a transaction - the multi-file replacement is not atomic and does not claim to be.
// A manifest with no hash fields is a voice written by an older build: accepted unverified,
// and upgraded in place (hashes of the payloads as they are) before the next replacement,
// so that window closes on first write.

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
    std::string ref_text;          // the reference transcript; with ref_codes, this is an ICL voice
    std::string model;             // model id the embedding was computed with
    std::string sample_sha256;     // of the stored sample bytes
    std::string created;           // ISO 8601 UTC
    double sample_seconds = 0;
    std::vector<float> embedding;
    std::vector<std::int32_t> ref_codes;   // [ref_frames][16]; empty = embedding-only cloning
    int ref_frames = 0;

    // Whether this voice can be used for in-context cloning: it needs both halves.
    bool icl() const { return ref_frames > 0 && !ref_codes.empty() && !ref_text.empty(); }
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

    // Whether a stored voice already is what a clone of this sample with this transcript
    // would produce - the question "is this registration a no-op?". The sample's bytes are
    // not enough: a voice registered before its transcript was known has to be re-cloned to
    // gain in-context cloning. `can_make_codes` is about the loaded model, and it only ever
    // makes this *more* permissive: a voice that already has verified codes stays current on
    // a model that cannot make them, because re-cloning there would throw them away.
    bool current(const std::string & id, const std::string & sample_sha256, const std::string & ref_text,
                 bool can_make_codes) const;

    // The verified codes of a stored voice whose sample is exactly these bytes, else empty.
    // Reference codes depend on the audio alone, so a clone that cannot make them - or that
    // only needs to change the transcript - inherits them instead of dropping them.
    std::vector<std::int32_t> codes_for_sample(const std::string & id, const std::string & sample_sha256,
                                               int * ref_frames) const;
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
