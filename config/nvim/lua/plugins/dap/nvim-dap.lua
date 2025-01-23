-- Dubger
-- https://github.com/mfussenegger/nvim-dap

-- Install Debug Adapter:
-- https://github.com/mfussenegger/nvim-dap/wiki/Debug-Adapter-installation

--
--[[
-- The gdb is a debugger and cppdbg is a debug adapter.
DAP-Client ----- Debug Adapter ------- Debugger ------ Debuggee
(nvim-dap)  |   (per language)  |   (per language)    (your app)
            |                   |
            |        Implementation specific communication
            |        Debug adapter and debugger could be the same process
            |
     Communication via the Debug Adapter Protocol
--]]

local Config = require("config")

return {
  "mfussenegger/nvim-dap",
  -- stylua: ignore
  keys = {
    { "<leader>dB", function() require("dap").set_breakpoint(vim.fn.input('Breakpoint condition: ')) end, desc = "Breakpoint Condition" },
    { "<leader>db", function() require("dap").toggle_breakpoint() end, desc = "Toggle Breakpoint" },
    { "<leader>dc", function() require("dap").continue() end, desc = "Continue" },
    { "<leader>dC", function() require("dap").run_to_cursor() end, desc = "Run to Cursor" },
    { "<leader>dg", function() require("dap").goto_() end, desc = "Go to line (no execute)" },
    { "<leader>di", function() require("dap").step_into() end, desc = "Step Into" },
    { "<leader>dj", function() require("dap").down() end, desc = "Down" },
    { "<leader>dk", function() require("dap").up() end, desc = "Up" },
    { "<leader>dl", function() require("dap").run_last() end, desc = "Run Last" },
    { "<leader>do", function() require("dap").step_out() end, desc = "Step Out" },
    { "<leader>dO", function() require("dap").step_over() end, desc = "Step Over" },
    { "<leader>dp", function() require("dap").pause() end, desc = "Pause" },
    { "<leader>dr", function() require("dap").repl.toggle() end, desc = "Toggle REPL" },
    { "<leader>ds", function() require("dap").session() end, desc = "Session" },
    { "<leader>dt", function() require("dap").terminate() end, desc = "Terminate" },
    { "<leader>dw", function() require("dap.ui.widgets").hover() end, desc = "Widgets" },
  },

  config = function()
    vim.api.nvim_set_hl(0, "DapStoppedLine", { default = true, link = "Visual" })

    for name, sign in pairs(Config.icons.dap) do
      sign = type(sign) == "table" and sign or { sign }
      vim.fn.sign_define(
        "Dap" .. name,
        { text = sign[1], texthl = sign[2] or "DiagnosticInfo", linehl = sign[3], numhl = sign[3] }
      )
    end

    -----
    ----- Config adapters.
    -----
    local dap = require("dap")
    dap.adapters.cppdbg = {
      -- For most debug adapters setting this is not necessary.
      id = "cppdbg",

      -- Either 'executable' or 'server'.
      -- `executable` to indicate that nvim-dap must launch the debug adapter.
      type = "executable",

      -- Command to invoke.
      command = "/root/.local/share/nvim/cppdbg/extension/debugAdapters/bin/OpenDebugAD7",
    }

    dap.adapters.codelldb = {
      -- `server`, to connect to a debug adapter via TCP.
      --        The adapter must be running, or started with a debug session
      --        via a `executable` configuration of the adapter.
      type = "server",
      port = "11534",
      executable = {
        -- CHANGE THIS to your path!
        command = "/root/.local/share/nvim/codelldb/extension/adapter/codelldb",
        args = { "--port", "11534" },
      },
    }

    dap.configurations.cpp = {
      {
        -- Which debug adapter to use.
        -- The type here established the link to the
        -- adapter definition: `dap.adapters.cpp`
        type = "cppdbg",

        -- Either `attach` or `launch`. Indicates whether the
        -- debug adapter should launch a debuggee or attach to
        -- one that is already running.
        request = "launch",

        -- A user-readable name for the configuration.
        name = "Launch File(cppdbg)",

        -- The absolute file to be debug.
        program = function()
          local default_path = vim.fn.getcwd() .. "/build/"
          local prompt = "Path to executable: " .. default_path
          return default_path .. vim.fn.input(prompt)
        end,

        cwd = "${workspaceFolder}",

        -- Stop at the main function.
        stopAtEntry = false,

        -- Some versions of GDB ship with pretty-printing support,
        -- which eases the debugging of strings and STL containers.
        setupCommands = {
          {
            text = "-enable-pretty-printing",
            description = "enable pretty printing",
            ignoreFailures = false,
          },
        },
      },
      {
        name = "Launch File(codelldb)",
        type = "codelldb",
        program = function()
          local default_path = vim.fn.getcwd() .. "/build/"
          local prompt = "Path to executable: " .. default_path
          return default_path .. vim.fn.input(prompt)
        end,
        request = "launch",
        cwd = "${workspaceFolder}",
        stopOnEntry = false,
      },
    }

    -- Reuse.
    dap.configurations.c = dap.configurations.cpp
  end,
}
