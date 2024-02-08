local M = {}

M.timer = vim.uv.new_timer()

function M.milliseconds_to_next_hour(millis)
  local seconds = millis / 1000
  local date = os.date("*t", seconds)
  local next_hour = (date.hour < 23) and date.hour + 1 or 0
  local next_minute = 0
  local next_second = 0
  local next_hour_seconds = os.time({
    year = date.year,
    month = date.month,
    day = date.day,
    hour = next_hour,
    min = next_minute,
    sec = next_second,
  })

  local seconds_to_next_hour = next_hour_seconds - seconds
  local milliseconds_to_next_hour = seconds_to_next_hour * 1000
  return milliseconds_to_next_hour
end

function M.message()
  return "one hour"
end

function M.start_report()
  local cur_time_ms = os.time() * 1000
  local timeout = M.milliseconds_to_next_hour(cur_time_ms)
  M.timer:start(timeout, 3600000, function()
    vim.notify(M.message(), vim.log.levels.INFO)
  end)
end

return M
