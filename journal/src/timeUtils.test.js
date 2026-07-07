import { formatCountdown, getRemainingSeconds } from './timeUtils';

describe('time countdown helpers', () => {
  test('formats seconds into hh:mm:ss', () => {
    expect(formatCountdown(3661)).toBe('01:01:01');
    expect(formatCountdown(59)).toBe('00:00:59');
    expect(formatCountdown(0)).toBe('00:00:00');
  });

  test('returns zero once the cooldown window has elapsed', () => {
    expect(getRemainingSeconds(25 * 60 * 60)).toBe(0);
    expect(getRemainingSeconds(24 * 60 * 60)).toBe(0);
    expect(getRemainingSeconds(23 * 60 * 60)).toBe(60 * 60);
  });
});
