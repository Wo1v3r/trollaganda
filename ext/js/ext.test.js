describe('Trollaganda Extension', () => {
  let uut;
  let chrome = {
    extension: {
      sendMessage: jest.fn()
    }
  };

  beforeEach(() => {
    global.chrome = chrome;
    uut = require('../src/inject/inject');
  });

  it('should call trollornot', () => {
    trollOrNot = jest.fn();
    uut.pollTrollOrNot();

    expect(trollOrNot).toBeCalled();
  });
});
