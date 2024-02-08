------------------------------------------------------
---        icons used by other plugins             ---
------------------------------------------------------
-- You can safely replace these icons to make a fancy neovim.
-- Fancy icon resources:
--   1. nerdfonts. Use cheat-sheet to select what icon you want,
--                 Then move cursor above the icon and choose copy icon.
--                 https://www.nerdfonts.com/cheat-sheet

-- Default icons used in DailyVim.
local icons = {
  -- Used for debug adapter
  dap = {
    Stopped = { "󰁕 ", "DiagnosticWarn", "DapStoppedLine" },
    Breakpoint = " ",
    BreakpointCondition = " ",
    BreakpointRejected = { " ", "DiagnosticError" },
    LogPoint = ".>",
  },

  -- Used for diagnostics.
  diagnostics = {
    Error = " ",
    Warn = " ",
    Hint = " ",
    Info = " ",
  },

  -- Used for git.
  git = {
    added = " ",
    modified = " ",
    removed = " ",
  },

  -- Used for Snippet.
  kinds = {
    Array = " ",
    Boolean = " ",
    Class = " ",
    Color = " ",
    Constant = " ",
    Constructor = " ",
    Copilot = " ",
    Enum = " ",
    EnumMember = " ",
    Event = " ",
    Field = " ",
    File = " ",
    Folder = " ",
    Function = " ",
    Interface = " ",
    Key = " ",
    Keyword = " ",
    Method = " ",
    Module = " ",
    Namespace = " ",
    Null = " ",
    Number = " ",
    Object = " ",
    Operator = " ",
    Package = " ",
    Property = " ",
    Reference = " ",
    Snippet = " ",
    String = " ",
    Struct = " ",
    Text = " ",
    TypeParameter = " ",
    Unit = " ",
    Value = " ",
    Variable = " ",
  },
}

return icons
