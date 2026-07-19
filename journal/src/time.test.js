const { hrsToHMS } = require('./time');

describe('time helpers', () => {
  test('formats hours into hh:mm:ss', () => {
    expect(hrsToHMS(1)).toBe('01:00:00');
    expect(hrsToHMS(0.5)).toBe('00:30:00');
  });
});
