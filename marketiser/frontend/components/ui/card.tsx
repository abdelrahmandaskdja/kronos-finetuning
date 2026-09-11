import type { HTMLAttributes, PropsWithChildren } from "react";

import clsx from "clsx";

export function Card({ children, className, ...props }: PropsWithChildren<HTMLAttributes<HTMLDivElement>>) {
  return (
    <div className={clsx("glass-panel rounded-3xl", className)} {...props}>
      {children}
    </div>
  );
}

