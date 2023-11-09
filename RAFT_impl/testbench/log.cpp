#include "include/log.hpp"

std::string formatString(std::string header, int color, std::string format) {
    std::stringstream ss;
    if (header.length() < 5){
        header = std::string(" ") + header;
    }
    ss << "[" << header << "] " << format;
    format = ss.str();
    ss.str("");
    if (color != 0){
        ss << "\033[" << std::to_string(color) << "m" << format << "\033[0m\n";
    }
    else {
        ss << format << std::endl;
    }
    return ss.str();
}

void LogInfo(std::string header, int color, std::string format) {
    std::cout << formatString(header, color, format);
}
