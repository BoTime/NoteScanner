'use client';

import { useEffect, useRef, useState } from 'react';
import { nextFocusIndex } from './popup-keys';

export interface PopupItem {
  key: string;
  label: string;
  disabled?: boolean;
  onSelect: () => void;
  /** Hover/focus preview hooks. The viewer uses them to drive `hoveredId`,
   *  exactly as the Radix menu items did. */
  onFocus?: () => void;
  onPointerEnter?: () => void;
  onPointerLeave?: () => void;
}

export interface PopupProps {
  x: number;
  y: number;
  items: PopupItem[];
  onClose: () => void;
}

export function Popup({ x, y, items, onClose }: PopupProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [focus, setFocus] = useState(0);

  useEffect(() => {
    function onDocPointerDown(e: PointerEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }
      const next = nextFocusIndex(focus, items.length, e.key);
      if (next !== focus) {
        e.preventDefault();
        setFocus(next);
      }
    }
    document.addEventListener('pointerdown', onDocPointerDown, true);
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('pointerdown', onDocPointerDown, true);
      document.removeEventListener('keydown', onKeyDown, true);
    };
  }, [focus, items.length, onClose]);

  useEffect(() => {
    const el = ref.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]')[focus];
    el?.focus();
  }, [focus]);

  return (
    <div
      ref={ref}
      className="sv-popup"
      role="menu"
      style={{ left: x, top: y }}
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item, i) => (
        <button
          key={item.key}
          type="button"
          role="menuitem"
          className="sv-popup-item"
          tabIndex={i === focus ? 0 : -1}
          disabled={item.disabled}
          onFocus={item.onFocus}
          onPointerEnter={item.onPointerEnter}
          onPointerLeave={item.onPointerLeave}
          onClick={() => {
            item.onSelect();
            onClose();
          }}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
