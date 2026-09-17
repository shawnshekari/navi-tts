#pragma once

// Minimal JSON: parse a document into a value tree, serialise back. Enough for
// tokenizer files and request bodies; no dependency. Throws navi::Error.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace navi {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array  = std::vector<Json>;
    using Object = std::map<std::string, Json>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : type_(Type::Bool), b_(b) {}
    Json(double d) : type_(Type::Number), num_(d) {}
    Json(int i) : type_(Type::Number), num_(i) {}
    Json(std::int64_t i) : type_(Type::Number), num_(static_cast<double>(i)) {}
    Json(std::string s) : type_(Type::String), str_(std::move(s)) {}
    Json(const char * s) : type_(Type::String), str_(s) {}
    Json(Array a) : type_(Type::Array), arr_(std::make_shared<Array>(std::move(a))) {}
    Json(Object o) : type_(Type::Object), obj_(std::make_shared<Object>(std::move(o))) {}

    static Json parse(std::string_view text);
    std::string dump() const;

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool                as_bool() const;
    double              as_number() const;
    std::int64_t        as_int() const;
    const std::string & as_string() const;
    const Array &       as_array() const;
    const Object &      as_object() const;

    // object access; get() throws on a missing key, find() returns nullptr
    const Json & get(std::string_view key) const;
    const Json * find(std::string_view key) const;
    bool         has(std::string_view key) const { return find(key) != nullptr; }
    Json &       set(std::string key, Json v);   // makes this an object if null

    // typed access with default
    std::string  str_or(std::string_view key, std::string dflt) const;
    double       num_or(std::string_view key, double dflt) const;
    std::int64_t int_or(std::string_view key, std::int64_t dflt) const;
    bool         bool_or(std::string_view key, bool dflt) const;

private:
    Type type_ = Type::Null;
    bool b_ = false;
    double num_ = 0.0;
    std::string str_;
    std::shared_ptr<Array> arr_;
    std::shared_ptr<Object> obj_;
};

} // namespace navi
