import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Check, ChevronDown, Database } from "lucide-react";
import type { Dataset } from "../../types";

interface DatasetSelectorProps {
  datasets: Dataset[];
  selected?: Dataset;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (dataset: Dataset) => void;
}

const formatNumber = (value: number) => new Intl.NumberFormat("en-US").format(value);

export function DatasetSelector({
  datasets,
  selected,
  open,
  onOpenChange,
  onSelect,
}: DatasetSelectorProps) {
  return (
    <DropdownMenu.Root open={open} onOpenChange={onOpenChange}>
      <DropdownMenu.Trigger asChild>
        <button className="dataset-trigger" aria-label="Dataset selector">
          <span className="dataset-trigger-icon"><Database aria-hidden="true" size={16} /></span>
          <span className="dataset-trigger-copy">
            <strong>{selected?.name ?? "Choose dataset"}</strong>
            <small>
              {selected
                ? `${formatNumber(selected.observations)} observations · ${selected.alternatives.length} alternatives`
                : "Mock datasets"}
            </small>
          </span>
          <ChevronDown aria-hidden="true" size={15} />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="dropdown-content dataset-menu" sideOffset={8} align="start">
          <div className="dropdown-label">Available datasets</div>
          {datasets.map((dataset) => (
            <DropdownMenu.Item
              key={dataset.id}
              className="dataset-option"
              onSelect={() => onSelect(dataset)}
            >
              <span className="dataset-check">
                {selected?.id === dataset.id && <Check aria-hidden="true" size={15} />}
              </span>
              <span>
                <strong>{dataset.name}</strong>
                <small>
                  {formatNumber(dataset.observations)} observations · {dataset.alternatives.length} alternatives
                </small>
                <em>{dataset.description}</em>
              </span>
            </DropdownMenu.Item>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
