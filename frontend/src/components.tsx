import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
}

export function Card({ children, className = "" }: CardProps) {
  return <section className={`card ${className}`}>{children}</section>;
}

interface StatProps {
  label: string;
  value: string;
  unit?: string;
  source?: string;
}

export function Stat({ label, value, unit, source }: StatProps) {
  return (
    <Card className="stat-card">
      <span className="stat-label">{label}</span>
      <strong className="stat-value">
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </strong>
      {source && <span className="stat-source">{source}</span>}
    </Card>
  );
}

interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
  variant?: "primary" | "secondary";
  disabled?: boolean;
}

export function Button({
  children,
  onClick,
  type = "button",
  variant = "primary",
  disabled = false,
}: ButtonProps) {
  return (
    <button
      className={`button button-${variant}`}
      type={type}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

interface TabsProps<T extends string> {
  items: Array<{ id: T; label: string }>;
  selected: T;
  onChange: (id: T) => void;
}

export function Tabs<T extends string>({
  items,
  selected,
  onChange,
}: TabsProps<T>) {
  return (
    <div className="tabs" role="tablist">
      {items.map((item, index) => (
        <button
          key={item.id}
          role="tab"
          aria-selected={selected === item.id}
          tabIndex={selected === item.id ? 0 : -1}
          className={`tab ${selected === item.id ? "tab-selected" : ""}`}
          onClick={() => onChange(item.id)}
          onKeyDown={(event) => {
            let nextIndex = index;
            if (event.key === "ArrowRight")
              nextIndex = (index + 1) % items.length;
            else if (event.key === "ArrowLeft") {
              nextIndex = (index - 1 + items.length) % items.length;
            } else if (event.key === "Home") nextIndex = 0;
            else if (event.key === "End") nextIndex = items.length - 1;
            else return;
            event.preventDefault();
            const nextTab =
              event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>(
                '[role="tab"]',
              )[nextIndex];
            nextTab?.focus();
            onChange(items[nextIndex].id);
          }}
          type="button"
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  unit = "",
  onChange,
}: SliderProps) {
  const id = `slider-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  return (
    <label className="slider-control" htmlFor={id}>
      <span className="slider-heading">
        <span>{label}</span>
        <strong>
          {value}
          {unit && ` ${unit}`}
        </strong>
      </span>
      <input
        id={id}
        aria-label={label}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

interface DropzoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
  title?: string;
  inputLabel?: string;
  hint?: string;
}

export function Dropzone({
  onFile,
  disabled = false,
  title = "Déposez un fichier NIfTI ici",
  inputLabel = "Sélectionner un fichier NIfTI",
  hint = "ou sélectionnez un fichier .nii / .nii.gz (max. 300 Mo)",
}: DropzoneProps) {
  const handleFile = (file: File | undefined) => {
    if (file) onFile(file);
  };
  return (
    <label
      className={`dropzone ${disabled ? "dropzone-disabled" : ""}`}
      role="button"
      aria-label={inputLabel}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        handleFile(event.dataTransfer.files[0]);
      }}
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.currentTarget.querySelector("input")?.click();
        }
      }}
    >
      <span className="dropzone-mark" aria-hidden="true">
        +
      </span>
      <strong>{title}</strong>
      <span>{hint}</span>
      <input
        aria-label={inputLabel}
        type="file"
        accept=".nii,.nii.gz"
        disabled={disabled}
        onChange={(event) => handleFile(event.target.files?.[0])}
      />
    </label>
  );
}

export interface DataColumn<T> {
  key: string;
  label: string;
  render: (row: T) => ReactNode;
}

interface DataTableProps<T> {
  columns: DataColumn<T>[];
  rows: T[];
  emptyMessage: string;
}

export function DataTable<T>({
  columns,
  rows,
  emptyMessage,
}: DataTableProps<T>) {
  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} scope="col">
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="table-empty">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            rows.map((row, index) => (
              <tr key={index}>
                {columns.map((column) => (
                  <td key={column.key}>{column.render(row)}</td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

interface BannerProps {
  children: ReactNode;
  tone?: "warning" | "error" | "info";
}

export function Banner({ children, tone = "info" }: BannerProps) {
  return (
    <div className={`banner banner-${tone}`} role="status">
      {children}
    </div>
  );
}

interface ChartFrameProps {
  title: string;
  caption: string;
  source: string;
  children: ReactNode;
  className?: string;
}

export function ChartFrame({
  title,
  caption,
  source,
  children,
  className = "",
}: ChartFrameProps) {
  return (
    <Card className={`chart-frame ${className}`}>
      <div className="chart-heading">
        <div>
          <h2>{title}</h2>
          <p>{caption}</p>
        </div>
        <span className="chart-source">{source}</span>
      </div>
      <div className="chart-body">{children}</div>
    </Card>
  );
}
