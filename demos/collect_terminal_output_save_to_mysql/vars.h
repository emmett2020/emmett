#pragma once

#define FMT_HEADER_ONLY
#include <spdlog/sinks/daily_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <vector>
using json = nlohmann::json;

template <typename T>
struct InnerMsg {
  const int MSG_TYPE_USER_COMMAND = 0;
  const int MSG_TYPE_CHILD_PROCESS_RET = 1;

  InnerMsg(int type, T data) : m_type(type), m_data(data) {}

  int get_msg_type() { return m_type; }

  T get_msg_data() { return m_data; }

 private:
  int m_type;
  T m_data;
};

const int MQ_MSG_MAX_SIZE = 1000;
const int MQ_MSG_NUM = 100;

struct LSResStruct {
  std::string perm;
  std::string cnt;
  std::string uid;
  std::string gid;
  std::string size;
  std::string data;
  std::string name;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LSResStruct, perm, cnt, uid, gid, size, data,
                                 name);
};
