#include <iostream>
#include <string>
#include <typeinfo>

std::string collectInfo(const void* obj, const std::string& typeName) {
    std::string json = "{";

    json += "\"type_name\":\"" + typeName + "\",";
    json += "\"pointer_address\":\"" + std::to_string((uintptr_t)obj) + "\"";

    json += "}";
    return json;
}
