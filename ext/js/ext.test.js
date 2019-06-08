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

  it('should poll once per second', async () => {
    global.fetch = jest
      .fn()
      .mockImplementation((_, { body }) =>
        Promise.resolve({ json: () => JSON.parse(body).messages.map(message => !!message) })
      );

    uut.pollTrollOrNot();

    await new Promise(resolve => setTimeout(resolve, 3500));

    expect(global.fetch).toHaveBeenCalledTimes(3);
  });

  it('should find message nodes in a document', () => {
    jest.spyOn(document, 'querySelectorAll').mockImplementation(selector => {
      if (selector === 'li [data-item-id]') {
        return [{}, {}, {}];
      }
    });

    const nodes = uut.getPageMessageNodes();

    expect(nodes.length).toBe(3);
  });

  it('should filter out nodes with no text container', () => {
    const nodeWithNoTextContainer = {
      querySelector: selector => (selector !== '.js-tweet-text-container' ? {} : undefined)
    };

    const nodeWithTextContainer = {
      querySelector: selector => (selector === '.js-tweet-text-container' ? {} : undefined)
    };

    expect(uut.withTextContainer(nodeWithNoTextContainer)).toBe(false);
    expect(uut.withTextContainer(nodeWithTextContainer)).toBe(true);
  });

  it('should map node to message dto', () => {
    const node = {
      attributes: {
        'data-item-id': { value: '1' }
      },
      querySelector: selector =>
        selector === '.js-tweet-text-container' ? { innerText: 'text' } : undefined
    };

    expect(uut.toMessages(node)).toStrictEqual({ id: '1', text: 'text' });
  });

  it('should extract text from a message DTO', () => {
    const extractedText = uut.extractText({ id: '1', text: 'text' });

    expect(extractedText).toBe('text');
  });

  it('should fetch predictions from the server', async () => {
    global.fetch = jest
      .fn()
      .mockImplementation((_, { body }) =>
        Promise.resolve({ json: () => JSON.parse(body).messages.map(message => !!message) })
      );

    const predictions = await uut.getPredictions([{ text: 'message1' }, { text: 'message2' }]);

    expect(fetch).toBeCalledWith('http://localhost:9090/predict', {
      method: 'POST',
      body: JSON.stringify({ messages: ['message1', 'message2'] }),
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    });

    expect(predictions).toStrictEqual([true, true]);
  });

  it('should color a tweet by its prediction', () => {
    const colorSpy = jest.fn();
    class StyleMock {
      set backgroundColor(color) {
        colorSpy(color);
      }
    }

    jest.spyOn(document, 'querySelector').mockImplementation(selector => {
      if (selector === 'li [data-item-id="messageID"]') {
        return {
          style: new StyleMock()
        };
      }
    });

    uut.colorTweet({ id: 'messageID', prediction: true });
    expect(colorSpy).toBeCalledWith('red');

    uut.colorTweet({ id: 'messageID', prediction: false });
  });
});
