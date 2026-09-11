import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

import clsx from "clsx";

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-[linear-gradient(135deg,rgba(30,216,193,1),rgba(18,184,166,0.86))] text-slate-950 hover:brightness-110",
  secondary:
    "border border-[var(--panel-border)] bg-white/6 text-white hover:border-white/20 hover:bg-white/10",
  ghost:
    "border border-transparent bg-transparent text-[var(--muted)] hover:border-[var(--panel-border)] hover:bg-white/6 hover:text-white",
  danger:
    "border border-red-500/20 bg-red-500/10 text-red-100 hover:bg-red-500/18",
};

const sizeClasses: Record<ButtonSize, string> = {
  sm: "h-9 rounded-2xl px-3.5 text-sm",
  md: "h-11 rounded-2xl px-4 text-sm",
  lg: "h-13 rounded-2xl px-5 text-sm uppercase tracking-[0.22em]",
};

export function Button({
  children,
  className,
  variant = "primary",
  size = "md",
  type = "button",
  ...props
}: PropsWithChildren<ButtonProps>) {
  return (
    <button
      type={type}
      className={clsx(
        "inline-flex items-center justify-center gap-2 font-medium transition-all disabled:cursor-not-allowed disabled:opacity-50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

