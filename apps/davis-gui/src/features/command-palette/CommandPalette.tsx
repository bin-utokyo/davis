import * as Dialog from "@radix-ui/react-dialog";
import {
  BarChart3,
  Database,
  FileSpreadsheet,
  FlaskConical,
  FolderOpen,
  GitCompareArrows,
  Search,
} from "lucide-react";
import { useMemo, useState } from "react";

export type CommandId =
  | "new-model"
  | "choose-dataset"
  | "open-experiment"
  | "compare-models"
  | "estimation-table"
  | "plot-coefficients";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCommand: (command: CommandId) => void;
}

const commands = [
  {
    id: "new-model" as const,
    label: "New MNL Model",
    description: "Start from the guided base specification",
    icon: FlaskConical,
    group: "Model",
  },
  {
    id: "choose-dataset" as const,
    label: "Choose Dataset",
    description: "Switch the mock estimation sample",
    icon: Database,
    group: "Data",
  },
  {
    id: "open-experiment" as const,
    label: "Open Experiment",
    description: "Jump to a saved specification",
    icon: FolderOpen,
    group: "Model",
  },
  {
    id: "compare-models" as const,
    label: "Compare Models",
    description: "Review fit metrics side by side",
    icon: GitCompareArrows,
    group: "Analysis",
  },
  {
    id: "estimation-table" as const,
    label: "Generate Estimation Table",
    description: "Open the parameter table",
    icon: FileSpreadsheet,
    group: "Results",
  },
  {
    id: "plot-coefficients" as const,
    label: "Plot Coefficients",
    description: "Open the confidence interval view",
    icon: BarChart3,
    group: "Results",
  },
];

export function CommandPalette({ open, onOpenChange, onCommand }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);

  const filteredCommands = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized) return commands;
    return commands.filter((command) =>
      `${command.label} ${command.description} ${command.group}`
        .toLowerCase()
        .includes(normalized),
    );
  }, [query]);

  const selectCommand = (command: CommandId) => {
    onCommand(command);
    onOpenChange(false);
    setQuery("");
  };

  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="command-dialog" aria-describedby="command-help">
          <Dialog.Title className="sr-only">DAVIS command palette</Dialog.Title>
          <Dialog.Description id="command-help" className="sr-only">
            Search commands. Use the arrow keys to move and Enter to select.
          </Dialog.Description>
          <div className="command-input-row">
            <Search aria-hidden="true" size={19} />
            <input
              autoFocus
              className="command-input"
              aria-label="Search DAVIS commands"
              aria-controls="command-list"
              aria-activedescendant={
                filteredCommands[activeIndex]
                  ? `command-${filteredCommands[activeIndex].id}`
                  : undefined
              }
              role="combobox"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setActiveIndex(0);
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setActiveIndex((index) =>
                    filteredCommands.length ? (index + 1) % filteredCommands.length : 0,
                  );
                }
                if (event.key === "ArrowUp") {
                  event.preventDefault();
                  setActiveIndex((index) =>
                    filteredCommands.length
                      ? (index - 1 + filteredCommands.length) % filteredCommands.length
                      : 0,
                  );
                }
                if (event.key === "Enter" && filteredCommands[activeIndex]) {
                  event.preventDefault();
                  selectCommand(filteredCommands[activeIndex].id);
                }
              }}
              placeholder="Search DAVIS or enter a command..."
            />
            <kbd>ESC</kbd>
          </div>
          <div id="command-list" className="command-list" role="listbox">
            {filteredCommands.length ? (
              filteredCommands.map((command, index) => {
                const Icon = command.icon;
                return (
                  <button
                    key={command.id}
                    id={`command-${command.id}`}
                    className="command-option"
                    data-active={activeIndex === index}
                    role="option"
                    aria-selected={activeIndex === index}
                    onMouseMove={() => setActiveIndex(index)}
                    onClick={() => selectCommand(command.id)}
                  >
                    <span className="command-icon">
                      <Icon aria-hidden="true" size={18} />
                    </span>
                    <span className="command-copy">
                      <strong>{command.label}</strong>
                      <small>{command.description}</small>
                    </span>
                    <span className="command-group">{command.group}</span>
                  </button>
                );
              })
            ) : (
              <div className="command-empty">No commands found</div>
            )}
          </div>
          <div className="command-footer">
            <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
            <span><kbd>↵</kbd> Select</span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
