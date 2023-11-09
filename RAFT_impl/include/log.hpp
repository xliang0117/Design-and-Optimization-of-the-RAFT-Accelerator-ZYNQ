/**===============LOG模块==============
 * @author dxd
 * @date 2021.9.28
 * @note 输出彩色 + header + 局部等级管理
 */

#ifndef LOGDXD_H
#define LOGDXD_H

#pragma once
#include <iostream>
#include <sstream>

// do not define LOCAL_LOG_LEVEL in header file!
enum LOG_LEVEL_TYPE {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_NONE
};

template<typename T>
std::string toString(T data) {
    std::stringstream ss;
    ss << data;
    return ss.str();
}

std::string formatString(std::string header, int color, std::string format);

void LogInfo(std::string header, int color, std::string format);

template<typename... types>
static void LogInfo(std::string header, int color, std::string format, const types&... args){
    printf(formatString(header, color, format).c_str(), args...);
}

#define CHECK_LOG_LEVEL(a, b, c) (a <= b ? c : void())

#define LOG_DEBUG_S(format)         CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 0, LogInfo("DEBUG", 0, format))       // Characters have no color
#define LOG_INFO_S(format)          CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 1, LogInfo("INFO", 32, format))       // Characters are printed in green
#define LOG_WARN_S(format)          CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 2, LogInfo("WARN", 33, format))       // Characters are printed in yellow
#define LOG_ERROR_S(format)         CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 3, LogInfo("ERROR", 31, format))      // Characters are printed in red

#define LOG_DEBUG_VAR(x)            CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 0, LogInfo("DEBUG", 0, std::string(#x) + ": " + toString(x)))
#define LOG_INFO_VAR(x)             CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 1, LogInfo("INFO", 32, std::string(#x) + ": " + toString(x)))
#define LOG_WARN_VAR(x)             CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 2, LogInfo("WARN", 33, std::string(#x) + ": " + toString(x)))
#define LOG_ERROR_VAR(x)            CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 3, LogInfo("ERROR", 31, std::string(#x) + ": " + toString(x)))

#define LOG_DEBUG(format, args...)  CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 0, LogInfo("DEBUG", 0, format, args))
#define LOG_INFO(format, args...)   CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 1, LogInfo("INFO", 32, format, args))
#define LOG_WARN(format, args...)   CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 2, LogInfo("WARN", 33, format, args))
#define LOG_ERROR(format, args...)  CHECK_LOG_LEVEL(LOCAL_LOG_LEVEL, 3, LogInfo("ERROR", 31, format, args))

#endif