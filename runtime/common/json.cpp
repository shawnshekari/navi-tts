#include "runtime/common/json.h"

#include "runtime/common/error.h"
#include "runtime/text/unicode.h"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace navi {

namespace {

struct Parser {
    std::string_view s;
    std::size_t i = 0;
    int depth = 0;

    [[noreturn]] void err(const std::string & what) const {
        fail("json: " + what + " at offset " + std::to_string(i));
    }
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    bool eat(char c) { if (i < s.size() && s[i] == c) { ++i; return true; } return false; }
    void expect(char c) { if (!eat(c)) err(std::string("expected '") + c + "'"); }

    Json value() {
        ws();
        if (i >= s.size()) err("unexpected end");
        if (++depth > 256) err("nesting too deep");
        Json v;
        switch (s[i]) {
            case '{': v = object(); break;
            case '[': v = array(); break;
            case '"': v = Json(string()); break;
            case 't': lit("true"); v = Json(true); break;
            case 'f': lit("false"); v = Json(false); break;
            case 'n': lit("null"); v = Json(nullptr); break;
            default: v = number(); break;
        }
        --depth;
        return v;
    }
    void lit(std::string_view w) {
        if (s.substr(i, w.size()) != w) err("bad literal");
        i += w.size();
    }
    Json number() {
        const std::size_t start = i;
        if (eat('-')) {}
        while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' || s[i] == 'e' || s[i] == 'E' ||
                                s[i] == '+' || s[i] == '-')) ++i;
        double d = 0.0;
        const auto r = std::from_chars(s.data() + start, s.data() + i, d);
        if (r.ec != std::errc{} || r.ptr != s.data() + i) err("bad number");
        return Json(d);
    }
    unsigned hex4() {
        if (i + 4 > s.size()) err("bad \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else err("bad \\u escape");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string out;
        while (true) {
            if (i >= s.size()) err("unterminated string");
            const char c = s[i++];
            if (c == '"') break;
            if (static_cast<unsigned char>(c) < 0x20) err("control character in string");
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) err("bad escape");
            const char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    char32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                            i += 2;
                            const unsigned lo = hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) err("bad surrogate pair");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    text::utf8_append(out, cp);
                    break;
                }
                default: err("bad escape");
            }
        }
        return out;
    }
    Json array() {
        expect('[');
        Json::Array a;
        ws();
        if (eat(']')) return Json(std::move(a));
        while (true) {
            a.push_back(value());
            ws();
            if (eat(',')) continue;
            expect(']');
            return Json(std::move(a));
        }
    }
    Json object() {
        expect('{');
        Json::Object o;
        ws();
        if (eat('}')) return Json(std::move(o));
        while (true) {
            ws();
            std::string k = string();
            ws();
            expect(':');
            o[std::move(k)] = value();
            ws();
            if (eat(',')) continue;
            expect('}');
            return Json(std::move(o));
        }
    }
};

void dump_string(std::string & out, const std::string & s) {
    out += '"';
    for (const char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    out += '"';
}

void dump_into(std::string & out, const Json & v) {
    switch (v.type()) {
        case Json::Type::Null: out += "null"; break;
        case Json::Type::Bool: out += v.as_bool() ? "true" : "false"; break;
        case Json::Type::Number: {
            const double d = v.as_number();
            if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 1e15) {
                out += std::to_string(static_cast<long long>(d));
            } else {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.17g", d);
                out += buf;
            }
            break;
        }
        case Json::Type::String: dump_string(out, v.as_string()); break;
        case Json::Type::Array: {
            out += '[';
            bool first = true;
            for (const Json & e : v.as_array()) { if (!first) out += ','; first = false; dump_into(out, e); }
            out += ']';
            break;
        }
        case Json::Type::Object: {
            out += '{';
            bool first = true;
            for (const auto & [k, e] : v.as_object()) {
                if (!first) out += ',';
                first = false;
                dump_string(out, k);
                out += ':';
                dump_into(out, e);
            }
            out += '}';
            break;
        }
    }
}

const char * type_name(Json::Type t) {
    switch (t) {
        case Json::Type::Null: return "null";
        case Json::Type::Bool: return "bool";
        case Json::Type::Number: return "number";
        case Json::Type::String: return "string";
        case Json::Type::Array: return "array";
        case Json::Type::Object: return "object";
    }
    return "?";
}

[[noreturn]] void type_err(Json::Type have, const char * want) {
    fail(std::string("json: expected ") + want + ", have " + type_name(have));
}

} // namespace

Json Json::parse(std::string_view text) {
    Parser p{text};
    Json v = p.value();
    p.ws();
    if (p.i != text.size()) p.err("trailing characters");
    return v;
}

std::string Json::dump() const {
    std::string out;
    dump_into(out, *this);
    return out;
}

bool Json::as_bool() const { if (type_ != Type::Bool) type_err(type_, "bool"); return b_; }
double Json::as_number() const { if (type_ != Type::Number) type_err(type_, "number"); return num_; }
std::int64_t Json::as_int() const { return static_cast<std::int64_t>(as_number()); }
const std::string & Json::as_string() const { if (type_ != Type::String) type_err(type_, "string"); return str_; }
const Json::Array & Json::as_array() const { if (type_ != Type::Array) type_err(type_, "array"); return *arr_; }
const Json::Object & Json::as_object() const { if (type_ != Type::Object) type_err(type_, "object"); return *obj_; }

const Json * Json::find(std::string_view key) const {
    if (type_ != Type::Object) return nullptr;
    const auto it = obj_->find(std::string(key));
    return it == obj_->end() ? nullptr : &it->second;
}

const Json & Json::get(std::string_view key) const {
    const Json * v = find(key);
    if (!v) fail("json: missing key '" + std::string(key) + "'");
    return *v;
}

Json & Json::set(std::string key, Json v) {
    if (type_ == Type::Null) { type_ = Type::Object; obj_ = std::make_shared<Object>(); }
    if (type_ != Type::Object) type_err(type_, "object");
    return (*obj_)[std::move(key)] = std::move(v);
}

std::string Json::str_or(std::string_view key, std::string dflt) const {
    const Json * v = find(key);
    return (v && v->is_string()) ? v->as_string() : std::move(dflt);
}
double Json::num_or(std::string_view key, double dflt) const {
    const Json * v = find(key);
    return (v && v->is_number()) ? v->as_number() : dflt;
}
std::int64_t Json::int_or(std::string_view key, std::int64_t dflt) const {
    const Json * v = find(key);
    return (v && v->is_number()) ? v->as_int() : dflt;
}
bool Json::bool_or(std::string_view key, bool dflt) const {
    const Json * v = find(key);
    return (v && v->is_bool()) ? v->as_bool() : dflt;
}

} // namespace navi
