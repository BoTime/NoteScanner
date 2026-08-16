/** Roving focus for the menu popup. Wraps at both ends; unrelated keys hold. */
export function nextFocusIndex(current: number, count: number, key: string): number {
  if (count <= 0) return 0;
  switch (key) {
    case 'ArrowDown':
      return (current + 1) % count;
    case 'ArrowUp':
      return (current - 1 + count) % count;
    case 'Home':
      return 0;
    case 'End':
      return count - 1;
    default:
      return current;
  }
}
